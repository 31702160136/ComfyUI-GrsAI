"""
GrsAI API客户端
封装与grsai.com的所有交互逻辑
"""

import json
import time
import requests
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:
    from PIL import Image

try:
    from .config import GrsaiConfig, default_config
    from .utils import format_error_message, download_image
except ImportError:
    from config import GrsaiConfig, default_config
    from utils import format_error_message, download_image


class GrsaiAPIError(Exception):
    """API调用异常"""

    pass


class GrsaiAPI:
    """GrsAI API客户端类"""

    def __init__(self, api_key: str, config: Optional[GrsaiConfig] = None):
        """
        初始化API客户端

        Args:
            api_key: API密钥
            config: 配置对象
        """
        if not api_key or not api_key.strip():
            raise GrsaiAPIError("API密钥在初始化时不能为空")
        normalized_api_key = api_key.strip()
        if not normalized_api_key.startswith("sk-"):
            raise GrsaiAPIError(
                "API密钥格式无效：请输入以 sk- 开头的真实 GrsAI API Key，"
                "不要使用输入框中的中文占位文本"
            )
        try:
            normalized_api_key.encode("ascii")
        except UnicodeEncodeError as exc:
            raise GrsaiAPIError(
                "API密钥格式无效：密钥中包含中文或其他非 ASCII 字符，" "请求尚未提交"
            ) from exc

        self.api_key = normalized_api_key
        self.config = config or default_config
        self.session = requests.Session()
        self.last_task_id: Optional[str] = None
        self._setup_session()

    @staticmethod
    def _extract_error_detail(payload: Any) -> str:
        """从接口错误响应中提取最具体的可读失败原因。"""
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, dict):
            details: List[str] = []
            for key in ("error", "message", "msg", "detail", "reason"):
                if key in payload:
                    detail = GrsaiAPI._extract_error_detail(payload[key])
                    if detail and detail not in details:
                        details.append(detail)
            return "; ".join(details)
        if isinstance(payload, list):
            details = [GrsaiAPI._extract_error_detail(item) for item in payload]
            return "; ".join(detail for detail in details if detail)
        return str(payload).strip()

    def _setup_session(self):
        """设置HTTP会话"""
        self.session.headers.update(
            {
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "ComfyUI-GrsAI/1.0",
            }
        )

        # 直接使用传入的API密钥设置认证头
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        发送HTTP请求

        Args:
            method: HTTP方法
            endpoint: API端点
            data: JSON 请求数据
            params: URL 查询参数
            timeout: 超时时间

        Returns:
            Dict: API响应数据

        Raises:
            GrsaiAPIError: API调用失败
        """
        url = f"{self.config.get_config('api_base_url')}{endpoint}"
        timeout = timeout or self.config.get_config("request_timeout", 60)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            if method.upper() == "POST":
                response = self.session.post(
                    url,
                    json=data,
                    params=params,
                    timeout=timeout,
                    headers=headers,
                )
            else:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=timeout,
                    headers=headers,
                )

            # 检查HTTP状态码
            if 200 <= response.status_code < 300:
                json_data = (
                    response.text[6:]
                    if response.text.startswith("data: ")
                    else response.text
                )
                try:
                    result = json.loads(json_data)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise GrsaiAPIError("API返回了无法解析的JSON响应") from exc
                if not isinstance(result, dict):
                    raise GrsaiAPIError(
                        f"API响应格式错误: 期望字典，实际为 {type(result).__name__}"
                    )
                return result

            error_detail = ""
            try:
                error_detail = self._extract_error_detail(response.json())
            except (ValueError, TypeError):
                error_detail = response.text.strip()
            if len(error_detail) > 1000:
                error_detail = error_detail[:997] + "..."

            if response.status_code == 401:
                error_msg = error_detail or "API密钥无效或已过期"
            elif response.status_code == 429:
                error_msg = error_detail or "请求频率过高"
            elif response.status_code >= 500:
                error_msg = error_detail or "接口服务器内部错误"
            else:
                error_msg = error_detail or "接口未提供具体错误信息"
            raise GrsaiAPIError(f"API请求失败 ({response.status_code}): {error_msg}")

        except requests.exceptions.Timeout:
            raise GrsaiAPIError("请求超时，请检查网络连接")
        except requests.exceptions.ConnectionError:
            raise GrsaiAPIError("网络连接失败，请检查网络设置")
        except UnicodeEncodeError as exc:
            raise GrsaiAPIError(
                "请求未提交：请求头包含无法编码的字符，请检查 API Key 是否误填了中文"
            ) from exc
        except GrsaiAPIError:
            raise
        except Exception as e:
            raise GrsaiAPIError(format_error_message(e, "网络请求"))

    def _raise_for_task_failure(self, response: Dict[str, Any]) -> None:
        """将异步任务的失败状态转换为统一异常。"""
        status = str(response.get("status", "unknown"))
        task_id = response.get("id", "unknown")
        error = self._extract_error_detail(response) or "服务未提供错误详情"
        raise GrsaiAPIError(f"异步生成失败 (任务: {task_id}, 状态: {status}): {error}")

    def _wait_for_async_result(
        self, initial_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """轮询新版异步结果接口，直到任务成功、失败或超时。"""
        task_id = initial_response.get("id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise GrsaiAPIError(f"异步生成响应缺少有效任务ID: {initial_response}")

        response = initial_response
        poll_interval = max(0.5, float(self.config.get_config("poll_interval", 2.0)))
        generation_timeout = max(
            poll_interval,
            float(self.config.get_config("generation_timeout", 3600)),
        )
        deadline = time.monotonic() + generation_timeout
        waiting_statuses = {"running", "pending", "queued", "processing", "submitted"}
        failed_statuses = {"failed", "violation", "cancelled", "canceled"}
        last_progress = None

        while True:
            status = str(response.get("status", "")).lower()
            if status == "succeeded":
                return response
            if status in failed_statuses:
                self._raise_for_task_failure(response)
            if status not in waiting_statuses:
                raise GrsaiAPIError(
                    f"异步任务返回未知状态 (任务: {task_id}): {status or '空状态'}"
                )

            progress = response.get("progress")
            if progress is not None and progress != last_progress:
                print(f"⏳ 异步任务 {task_id} 生成进度: {progress}%")
                last_progress = progress

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GrsaiAPIError(
                    f"异步生成超时 (任务: {task_id}, 等待: {generation_timeout:g}秒)"
                )

            time.sleep(min(poll_interval, remaining))
            response = self._make_request(
                "GET",
                "/v1/api/result",
                params={"id": task_id},
            )

    @staticmethod
    def _extract_result_urls(response: Dict[str, Any]) -> List[str]:
        """从新版 results 数组中提取有效的媒体 URL。"""
        results = response.get("results")
        if not isinstance(results, list):
            raise GrsaiAPIError("异步任务成功，但响应中缺少 results 列表")

        urls = [
            item.get("url")
            for item in results
            if isinstance(item, dict)
            and isinstance(item.get("url"), str)
            and item["url"].startswith(("http://", "https://"))
        ]
        if not urls:
            raise GrsaiAPIError("异步任务成功，但 results 中没有有效媒体 URL")
        return urls

    def _download_images(
        self, urls: List[str]
    ) -> Tuple[List["Image.Image"], List[str], List[str]]:
        """并行下载结果图片，并保持 API 返回顺序。"""
        downloaded: Dict[int, Tuple["Image.Image", str]] = {}
        errors: List[str] = []
        timeout = self.config.get_config("timeout", 300)

        def download_one(index: int, url: str):
            image = download_image(url, timeout=timeout)
            if image is None:
                raise GrsaiAPIError("图像下载失败，可能是网络超时或服务异常")
            return index, image, url

        with ThreadPoolExecutor(max_workers=min(len(urls), 8)) as executor:
            futures = {
                executor.submit(download_one, index, url): (index, url)
                for index, url in enumerate(urls)
            }
            for future in as_completed(futures):
                index, url = futures[future]
                try:
                    _, image, image_url = future.result()
                    downloaded[index] = (image, image_url)
                except Exception as exc:
                    errors.append(f"图片下载失败: {url} ({exc})")

        ordered = [downloaded[index] for index in sorted(downloaded)]
        return (
            [item[0] for item in ordered],
            [item[1] for item in ordered],
            errors,
        )

    def _generate_async(
        self, payload: Dict[str, Any]
    ) -> Tuple[List["Image.Image"], List[str], List[str]]:
        """通过新版统一接口提交异步任务、轮询并下载结果。"""
        self.last_task_id = None
        async_payload = {
            key: value
            for key, value in payload.items()
            if value is not None and value != ""
        }
        async_payload["replyType"] = "async"
        model = str(async_payload.get("model", "unknown"))
        image_count = len(async_payload.get("images", []))
        task_id = "未创建"
        request_details = {
            key: value
            for key, value in async_payload.items()
            if key not in {"prompt", "images"}
        }

        print(
            f"🚀 提交新版异步图片生成 | 模型: {model} | "
            f"参考图: {image_count} 张 | 参数: {request_details}"
        )

        try:
            response = self._make_request(
                "POST",
                "/v1/api/generate",
                data=async_payload,
            )
            task_id = str(response.get("id", "未返回"))
            if task_id and task_id != "未返回":
                self.last_task_id = task_id
            print(
                f"📨 异步任务提交成功 | 任务: {task_id} | "
                f"状态: {response.get('status', 'unknown')}"
            )

            completed = self._wait_for_async_result(response)
            result_urls = self._extract_result_urls(completed)
            print(
                f"✅ 图片生成成功 | 模型: {model} | 任务: {task_id} | "
                f"结果: {len(result_urls)} 张 | 进度: {completed.get('progress', 100)}%"
            )
            for index, url in enumerate(result_urls, 1):
                print(f"   结果 {index}: {url}")

            images, image_urls, errors = self._download_images(result_urls)
            for error in errors:
                print(f"❌ {error}")

            if errors:
                print(
                    f"⚠️ 结果下载完成 | 任务: {task_id} | "
                    f"成功: {len(images)} 张 | 失败: {len(errors)} 张"
                )
            else:
                print(
                    f"✅ 结果下载成功 | 任务: {task_id} | " f"已下载: {len(images)} 张"
                )
            return images, image_urls, errors
        except Exception as exc:
            print(
                f"❌ 图片生成失败 | 模型: {model} | 任务: {task_id} | "
                f"错误详情: {exc}"
            )
            raise

    def gpt_image_generate_image(
        self,
        prompt: str,
        model: str = "gpt-image-2",
        aspect_ratio: Optional[str] = None,
        urls: Optional[List[str]] = None,
        quality: Optional[str] = None,
        background: Optional[str] = None,
        mask: Optional[str] = None,
    ) -> Tuple[List["Image.Image"], List[str], List[str]]:
        """使用新版统一异步接口生成 GPT Image 图片。"""
        if quality is None:
            vip_parameter_models = {
                "gpt-image-2-vip",
                "gpt-image-2.5-flare",
                "gpt-image-2.5-sunburst",
            }
            quality = "medium" if model in vip_parameter_models else "auto"
        payload = {
            "model": model,
            "prompt": prompt,
            "images": urls or [],
            "aspectRatio": aspect_ratio,
            "quality": quality,
            "background": background,
            "mask": mask,
        }
        try:
            return self._generate_async(payload)
        except Exception as e:
            if isinstance(e, GrsaiAPIError):
                raise
            raise GrsaiAPIError(format_error_message(e, "图像生成"))

    def banana_generate_image(
        self,
        prompt: str,
        model: str = "nano-banana-fast",
        urls: Optional[List[str]] = None,
        aspect_ratio: Optional[str] = None,
        image_size: Optional[str] = None,
    ) -> Tuple[List["Image.Image"], List[str], List[str]]:
        """
        Nano Banana API 调用

        Args:
            prompt: 编辑或生成描述。
            model: 使用的模型，默认 "nano-banana-fast"。
                   可选值："nano-banana-fast"、"nano-banana"、"nano-banana-pro"、"nano-banana-pro-vt"。
            urls: 可选的参考/输入图片 URL 列表（用于编辑场景）。
            image_size: 仅 nano-banana-pro / nano-banana-pro-vt 支持的输出尺寸，可选 "1K" | "2K" | "4K"。

        Returns:
            (pil_images, image_urls, errors)
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "images": urls or [],
        }

        if image_size:
            # ComfyUI 侧无法基于模型动态隐藏 imageSize 参数，因此在非支持模型上直接忽略该参数
            if default_config.nano_banana_model_supports_image_size(model):
                if not default_config.validate_nano_banana_image_size(image_size):
                    raise GrsaiAPIError(
                        f"不支持的 imageSize: {image_size}. 支持的选项: {', '.join(default_config.SUPPORTED_NANO_BANANA_SIZES)}"
                    )
                payload["imageSize"] = image_size

        if aspect_ratio:
            if not default_config.validate_nano_banana_aspect_ratio(aspect_ratio):
                raise GrsaiAPIError(
                    f"不支持的宽高比: {aspect_ratio}. 支持的选项: {', '.join(default_config.SUPPORTED_NANO_BANANA_AR)}"
                )
            payload["aspectRatio"] = aspect_ratio

        try:
            return self._generate_async(payload)
        except Exception as e:
            if isinstance(e, GrsaiAPIError):
                raise
            raise GrsaiAPIError(format_error_message(e, "Nano Banana 调用"))

    def minimax_h3_generate_video(
        self,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        duration: int,
        images: Optional[List[str]] = None,
        audios: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """提交 MiniMax H3 异步视频任务并返回生成结果 URL。"""
        self.last_task_id = None
        if not prompt or not prompt.strip():
            raise GrsaiAPIError("视频提示词不能为空")
        if aspect_ratio not in {"portrait", "landscape"}:
            raise GrsaiAPIError("视频比例仅支持 portrait 或 landscape")
        if resolution not in {"480p", "768p", "1080p"}:
            raise GrsaiAPIError("视频分辨率仅支持 480p、768p 或 1080p")
        if not 1 <= duration <= 15:
            raise GrsaiAPIError("视频时长必须在 1 到 15 秒之间")
        if resolution == "1080p" and duration > 10:
            raise GrsaiAPIError("1080p 视频时长不能超过 10 秒")
        if seed is not None and not 0 <= seed <= 9999999999:
            raise GrsaiAPIError("视频随机种子必须是最多 10 位的非负整数")

        image_inputs = images or []
        audio_inputs = audios or []
        if len(image_inputs) > 9:
            raise GrsaiAPIError("MiniMax H3 最多支持 9 张参考图")
        if len(audio_inputs) > 3:
            raise GrsaiAPIError("MiniMax H3 最多支持 3 段参考音频")

        payload: Dict[str, Any] = {
            "model": "minimax-h3",
            "prompt": prompt,
            "aspectRatio": aspect_ratio,
            "images": image_inputs,
            "audios": audio_inputs,
            "resolution": resolution,
            "duration": duration,
            "replyType": "async",
        }
        if seed is not None:
            payload["seed"] = seed

        task_id = "未创建"
        print(
            "🚀 提交 MiniMax H3 异步视频生成 | "
            f"比例: {aspect_ratio} | 分辨率: {resolution} | 时长: {duration} 秒 | "
            f"参考图: {len(image_inputs)} 张 | 参考音频: {len(audio_inputs)} 段"
        )

        try:
            response = self._make_request(
                "POST",
                "/v1/api/generate",
                data=payload,
            )
            task_id = str(response.get("id", "未返回"))
            if task_id and task_id != "未返回":
                self.last_task_id = task_id
            print(
                f"📨 视频任务提交成功 | 任务: {task_id} | "
                f"状态: {response.get('status', 'unknown')}"
            )

            completed = self._wait_for_async_result(response)
            video_urls = self._extract_result_urls(completed)
            print(
                f"✅ 视频生成成功 | 模型: minimax-h3 | 任务: {task_id} | "
                f"结果: {len(video_urls)} 个"
            )
            for index, url in enumerate(video_urls, 1):
                print(f"   视频 {index}: {url}")
            return video_urls
        except Exception as exc:
            print(
                f"❌ 视频生成失败 | 模型: minimax-h3 | 任务: {task_id} | "
                f"错误详情: {exc}"
            )
            if isinstance(exc, GrsaiAPIError):
                raise
            raise GrsaiAPIError(format_error_message(exc, "MiniMax H3 视频生成"))

    def flux_generate_image(
        self,
        prompt: str,
        model: str = "flux-kontext-pro",
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        urls: Optional[List[str]] = None,
        output_format: Optional[str] = None,
        safety_tolerance: Optional[int] = None,
        prompt_upsampling: Optional[bool] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple["Image.Image", str]:
        # 构建请求数据
        payload = {
            "model": model,
            "prompt": prompt,
            "images": urls or [],
        }

        # 动态添加所有非空的可选参数
        # 这种方式更简洁且易于维护
        optional_params = {
            "seed": seed,
            "aspectRatio": aspect_ratio,
            "output_format": output_format,
            "safetyTolerance": safety_tolerance,
            "promptUpsampling": prompt_upsampling,
            "guidance": guidance_scale,
            "steps": num_inference_steps,
        }

        for key, value in optional_params.items():
            # 只有当值不是None，或者对于字符串，不是空字符串时，才添加到payload
            if value is not None and value != "":
                payload[key] = value

        try:
            pil_images, image_urls, errors = self._generate_async(payload)
        except Exception as e:
            if isinstance(e, GrsaiAPIError):
                raise
            raise GrsaiAPIError(format_error_message(e, "图像生成"))
        if not pil_images:
            detail = "; ".join(errors) if errors else "没有可用结果"
            raise GrsaiAPIError(f"图像生成成功，但图片下载失败: {detail}")
        return pil_images[0], image_urls[0]

    def test_connection(self) -> bool:
        """
        测试API连接

        Returns:
            bool: 连接是否成功
        """
        try:
            # 尝试一个简单的请求来测试连接
            self.flux_generate_image("test", seed=1)
            return True
        except:
            return False

    def get_api_status(self) -> Dict[str, Any]:
        """
        获取API状态信息

        Returns:
            Dict: 状态信息
        """
        status = {
            "api_key_valid": bool(self.config.get_api_key()),
            "base_url": self.config.get_config("api_base_url"),
            "model": self.config.get_config("model"),
            "timeout": self.config.get_config("timeout"),
        }

        # 测试连接
        try:
            status["connection_ok"] = self.test_connection()
        except:
            status["connection_ok"] = False

        return status
