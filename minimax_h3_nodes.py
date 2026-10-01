"""
MiniMax H3 视频生成 ComfyUI 节点。
支持文生视频、参考图、参考音频和异步结果轮询。
"""

import base64
import io
import logging
import wave
from typing import Any, Dict, List

import requests
import torch

try:
    from .api_client import GrsaiAPI
    from .http_client import (
        create_http_session,
        describe_request_error,
        normalize_timeout,
    )
    from .utils import format_error_message, tensor_to_pil
except ImportError:
    from api_client import GrsaiAPI
    from http_client import (
        create_http_session,
        describe_request_error,
        normalize_timeout,
    )
    from utils import format_error_message, tensor_to_pil

try:
    from comfy_api.latest import VideoFromFile
except ImportError:
    try:
        from comfy_api.input_impl import VideoFromFile
    except ImportError:
        VideoFromFile = None


class SuppressHTTPLogs:
    """在上传输入和请求 API 时临时减少 HTTP 调试日志。"""

    def __init__(self):
        self.logger_names = ["httpx", "httpcore", "urllib3.connectionpool"]
        self.original_levels: Dict[str, int] = {}

    def __enter__(self):
        for logger_name in self.logger_names:
            logger = logging.getLogger(logger_name)
            self.original_levels[logger_name] = logger.level
            logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for logger_name, original_level in self.original_levels.items():
            logging.getLogger(logger_name).setLevel(original_level)


def _image_to_base64(image: torch.Tensor) -> str:
    """将 ComfyUI IMAGE 的第一帧编码为 PNG Base64。"""
    pil_images = tensor_to_pil(image)
    if not pil_images:
        raise ValueError("无法读取参考图")

    buffer = io.BytesIO()
    pil_images[0].save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _audio_to_base64(audio: Dict[str, Any]) -> str:
    """将 ComfyUI AUDIO 的第一个批次编码为 16-bit PCM WAV Base64。"""
    if not isinstance(audio, dict):
        raise ValueError("参考音频格式无效")

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or sample_rate is None:
        raise ValueError("参考音频缺少 waveform 或 sample_rate")

    samples = torch.as_tensor(waveform).detach().cpu().float()
    if samples.ndim == 3:
        samples = samples[0]
    elif samples.ndim == 1:
        samples = samples.unsqueeze(0)
    elif samples.ndim != 2:
        raise ValueError("参考音频 waveform 必须为 [B, C, T] 或 [C, T]")

    if samples.shape[-1] == 0:
        raise ValueError("参考音频为空")

    samples = torch.nan_to_num(samples).clamp(-1.0, 1.0)
    pcm = (
        (samples.transpose(0, 1).contiguous() * 32767.0)
        .round()
        .to(torch.int16)
        .numpy()
        .tobytes()
    )

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(samples.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _download_video(video_url: str) -> io.BytesIO:
    """下载生成结果，并返回可供 ComfyUI 延迟解码的视频缓冲区。"""
    session = create_http_session(headers={"User-Agent": "ComfyUI-GrsAI/1.1.5"})
    try:
        with session.get(
            video_url,
            stream=True,
            timeout=normalize_timeout((30, 300)),
        ) as response:
            response.raise_for_status()
            video_buffer = io.BytesIO()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    video_buffer.write(chunk)
    except requests.RequestException as exc:
        raise RuntimeError(describe_request_error(exc, "视频下载")) from exc
    finally:
        session.close()

    if video_buffer.tell() == 0:
        raise RuntimeError("视频下载失败: API 返回了空文件")
    video_buffer.seek(0)
    return video_buffer


class _GrsaiMiniMaxH3NodeBase:
    """GrsAI MiniMax H3 分辨率节点公共实现。"""

    FUNCTION = "execute"
    CATEGORY = "GrsAI/Minimax H3"
    RESOLUTION = "768p"
    MAX_DURATION = 15

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "电影级写实风格，镜头运动稳定流畅，主体外观前后一致，"
                            "环境音与画面自然同步。"
                        ),
                    },
                ),
                "apikey": ("STRING", {"default": "请输入您的APIKEY: sk-xxxxxxx"}),
                "model": (["minimax-h3"], {"default": "minimax-h3"}),
                "aspect_ratio": (
                    ["portrait", "landscape"],
                    {"default": "landscape"},
                ),
                "duration": (
                    "INT",
                    {
                        "default": min(10, cls.MAX_DURATION),
                        "min": 1,
                        "max": cls.MAX_DURATION,
                        "step": 1,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999999999,
                        "step": 1,
                        "control_after_generate": True,
                    },
                ),
            },
            "optional": {
                **{f"image_{index}": ("IMAGE",) for index in range(1, 10)},
                **{f"audio_{index}": ("AUDIO",) for index in range(1, 4)},
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "status", "api_task_ids")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @staticmethod
    def _error_result(message: str):
        if "接口任务ID:" not in message:
            message += "\n接口任务ID: 未创建（请求未成功提交）"
        print(f"MiniMax H3 节点执行错误: {message}")
        raise RuntimeError(message)

    def execute(self, **kwargs):
        prompt = str(kwargs.pop("prompt", "")).strip()
        apikey = str(kwargs.pop("apikey", "")).strip()
        model = kwargs.pop("model", "minimax-h3")
        aspect_ratio = kwargs.pop("aspect_ratio", "landscape")
        resolution = self.RESOLUTION
        duration = int(kwargs.pop("duration", 10))
        seed = int(kwargs.pop("seed", 0))

        if model != "minimax-h3":
            return self._error_result(f"不支持的模型: {model}")
        if not prompt:
            return self._error_result("提示词不能为空")
        if not 1 <= duration <= self.MAX_DURATION:
            return self._error_result(
                f"{resolution} 分辨率的视频时长必须在 1 到 "
                f"{self.MAX_DURATION} 秒之间"
            )

        image_inputs: List[torch.Tensor] = [
            kwargs[f"image_{index}"]
            for index in range(1, 10)
            if kwargs.get(f"image_{index}") is not None
        ]
        audio_inputs: List[Dict[str, Any]] = [
            kwargs[f"audio_{index}"]
            for index in range(1, 4)
            if kwargs.get(f"audio_{index}") is not None
        ]

        try:
            encoded_images = [_image_to_base64(image) for image in image_inputs]
        except Exception as exc:
            return self._error_result(f"参考图编码失败: {format_error_message(exc)}")

        try:
            encoded_audios = [_audio_to_base64(audio) for audio in audio_inputs]
        except Exception as exc:
            return self._error_result(f"参考音频编码失败: {format_error_message(exc)}")

        api_client = None
        try:
            with SuppressHTTPLogs():
                api_client = GrsaiAPI(api_key=apikey)
                video_urls = api_client.minimax_h3_generate_video(
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    duration=duration,
                    images=encoded_images,
                    audios=encoded_audios,
                    seed=seed,
                )
                task_id = api_client.last_task_id or ""
        except Exception as exc:
            task_id = api_client.last_task_id if api_client is not None else None
            display_task_id = task_id or "未创建（请求未提交）"
            task_note = f" [接口任务ID: {display_task_id}]"
            return self._error_result(
                f"MiniMax H3 API 调用失败: {format_error_message(exc)}{task_note}"
            )

        if not video_urls:
            return self._error_result("API 未返回可用的视频链接")

        video_url = video_urls[0]
        if VideoFromFile is None:
            return self._error_result(
                "当前 ComfyUI 不支持原生 VIDEO 输出，请更新 ComfyUI 后重试"
            )

        try:
            video = VideoFromFile(_download_video(video_url))
        except Exception as exc:
            return self._error_result(
                f"创建 ComfyUI VIDEO 失败: {format_error_message(exc)}"
            )

        status = (
            f"MiniMax H3 | 模型: {model} | {aspect_ratio} | {resolution} | "
            f"{duration} 秒 | 参考图: {len(encoded_images)} 张 | "
            f"参考音频: {len(encoded_audios)} 段 | 生成成功"
        )
        if task_id:
            status += f" | 接口任务ID: {task_id}"
        return {
            "ui": {"string": [status]},
            "result": (video, status, task_id),
        }


class GrsaiMiniMaxH3_480pNode(_GrsaiMiniMaxH3NodeBase):
    """MiniMax H3 480p 视频生成节点，最长支持 15 秒。"""

    RESOLUTION = "480p"
    MAX_DURATION = 15


class GrsaiMiniMaxH3_768pNode(_GrsaiMiniMaxH3NodeBase):
    """MiniMax H3 768p 视频生成节点，最长支持 15 秒。"""

    RESOLUTION = "768p"
    MAX_DURATION = 15


class GrsaiMiniMaxH3_1080pNode(_GrsaiMiniMaxH3NodeBase):
    """MiniMax H3 1080p 视频生成节点，最长支持 10 秒。"""

    RESOLUTION = "1080p"
    MAX_DURATION = 10


NODE_CLASS_MAPPINGS = {
    "Grsai_MiniMaxH3_480p": GrsaiMiniMaxH3_480pNode,
    "Grsai_MiniMaxH3_768p": GrsaiMiniMaxH3_768pNode,
    "Grsai_MiniMaxH3_1080p": GrsaiMiniMaxH3_1080pNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Grsai_MiniMaxH3_480p": "🎬 GrsAI MiniMax H3 - 480p",
    "Grsai_MiniMaxH3_768p": "🎬 GrsAI MiniMax H3 - 768p",
    "Grsai_MiniMaxH3_1080p": "🎬 GrsAI MiniMax H3 - 1080p",
}
