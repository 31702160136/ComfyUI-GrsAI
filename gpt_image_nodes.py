"""
ComfyUI节点实现
定义 GPT Image 图像生成节点（文生图 / 图生图 / 多图）
"""

import base64
import io
import logging
from typing import Any, Tuple, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .api_client import GrsaiAPI, GrsaiAPIError
    from .config import default_config
    from .utils import (
        pil_to_tensor,
        format_error_message,
        tensor_to_pil,
    )
except ImportError:
    from api_client import GrsaiAPI, GrsaiAPIError
    from config import default_config
    from utils import pil_to_tensor, format_error_message, tensor_to_pil


class SuppressFalLogs:
    """临时抑制HTTP相关的详细日志的上下文管理器"""

    def __init__(self):
        self.loggers_to_suppress = [
            "httpx",
            "httpcore",
            "urllib3.connectionpool",
        ]
        self.original_levels: Dict[str, int] = {}

    def __enter__(self):
        for logger_name in self.loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            self.original_levels[logger_name] = logger.level
            logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for logger_name, original_level in self.original_levels.items():
            logging.getLogger(logger_name).setLevel(original_level)


# gpt-image-2-vip 尺寸映射：显示标签 -> 实际发送给 API 的尺寸
# 格式：尺寸 (比例, K等级)，支持 1K / 2K / 4K
ASPECT_RATIO_VIP_MAP: Dict[str, str] = {
    "auto": "auto",
    # 1:1
    "1024x1024 (1:1, 1K)": "1024x1024",
    "2048x2048 (1:1, 2K)": "2048x2048",
    "2880x2880 (1:1, 4K)": "2880x2880",
    # 16:9
    "1280x720 (16:9, 1K)": "1280x720",
    "2048x1152 (16:9, 2K)": "2048x1152",
    "3840x2160 (16:9, 4K)": "3840x2160",
    # 9:16
    "720x1280 (9:16, 1K)": "720x1280",
    "1152x2048 (9:16, 2K)": "1152x2048",
    "2160x3840 (9:16, 4K)": "2160x3840",
    # 4:3
    "1152x864 (4:3, 1K)": "1152x864",
    "2304x1728 (4:3, 2K)": "2304x1728",
    "3264x2448 (4:3, 4K)": "3264x2448",
    # 3:4
    "864x1152 (3:4, 1K)": "864x1152",
    "1728x2304 (3:4, 2K)": "1728x2304",
    "2448x3264 (3:4, 4K)": "2448x3264",
    # 3:2
    "1536x1024 (3:2, 1K)": "1536x1024",
    "2048x1360 (3:2, 2K)": "2048x1360",
    "3504x2336 (3:2, 4K)": "3504x2336",
    # 2:3
    "1024x1536 (2:3, 1K)": "1024x1536",
    "1360x2048 (2:3, 2K)": "1360x2048",
    "2336x3504 (2:3, 4K)": "2336x3504",
    # 5:4
    "1120x896 (5:4, 1K)": "1120x896",
    "2240x1792 (5:4, 2K)": "2240x1792",
    "3200x2560 (5:4, 4K)": "3200x2560",
    # 4:5
    "896x1120 (4:5, 1K)": "896x1120",
    "1792x2240 (4:5, 2K)": "1792x2240",
    "2560x3200 (4:5, 4K)": "2560x3200",
    # 21:9
    "1456x624 (21:9, 1K)": "1456x624",
    "2912x1248 (21:9, 2K)": "2912x1248",
    "3840x1648 (21:9, 4K)": "3840x1648",
    # 9:21
    "624x1456 (9:21, 1K)": "624x1456",
    "1248x2912 (9:21, 2K)": "1248x2912",
    "1648x3840 (9:21, 4K)": "1648x3840",
    # 1:3
    "688x2048 (1:3, 2K)": "688x2048",
    "1280x3840 (1:3, 4K)": "1280x3840",
    # 3:1
    "2048x688 (3:1, 2K)": "2048x688",
    "3840x1280 (3:1, 4K)": "3840x1280",
    # 2:1
    "1536x768 (2:1, 1K)": "1536x768",
    "3072x1536 (2:1, 2K)": "3072x1536",
    "3840x1920 (2:1, 4K)": "3840x1920",
    # 1:2
    "768x1536 (1:2, 1K)": "768x1536",
    "1536x3072 (1:2, 2K)": "1536x3072",
    "1920x3840 (1:2, 4K)": "1920x3840",
}


# gpt-image-2 尺寸映射：显示标签 -> 实际发送给 API 的尺寸
ASPECT_RATIO_STD_MAP: Dict[str, str] = {
    "auto": "auto",
    "1024x1024 (1:1)": "1024x1024",
    "1672x941 (16:9)": "1672x941",
    "941x1672 (9:16)": "941x1672",
    "1443x1090 (4:3)": "1443x1090",
    "1090x1443 (3:4)": "1090x1443",
    "1536x1024 (3:2)": "1536x1024",
    "1024x1536 (2:3)": "1024x1536",
    "1408x1120 (5:4)": "1408x1120",
    "1120x1408 (4:5)": "1120x1408",
    "1920x832 (21:9)": "1920x832",
    "832x1920 (9:21)": "832x1920",
    "1792x896 (2:1)": "1792x896",
    "896x1792 (1:2)": "896x1792",
}


def _resolve_aspect_ratio(
    label: Optional[str], mapping: Dict[str, str]
) -> Optional[str]:
    """将下拉显示标签转换为实际发送给 API 的尺寸值。

    兼容旧值（直接传入纯尺寸字符串）以及 None。
    """
    if label is None:
        return None
    return mapping.get(label, label)


class GrsaiGPTImage_Node:
    """
    GPT Image 图像生成节点
    """

    FUNCTION = "execute"
    CATEGORY = "GrsAI/GPT Image"

    def _execute_generation(
        self,
        apikey: str,
        final_prompt: str,
        num_images: int,
        model: str,
        urls: list[str] = [],
        aspect_ratio: str = "auto",
        **kwargs,
    ) -> Tuple[List[Any], List[str], List[str]]:
        results_pil, result_urls, errors = [], [], []

        def generate_single_image():
            try:
                api_client = GrsaiAPI(api_key=apikey)
                api_params = {
                    "prompt": final_prompt,
                    "model": model,
                    "urls": urls,
                    "aspect_ratio": aspect_ratio,
                }
                api_params.update(kwargs)
                pil_imgs, img_urls, errs = api_client.gpt_image_generate_image(
                    **api_params
                )
                return pil_imgs, img_urls, errs
            except Exception as e:
                return e

        with ThreadPoolExecutor(max_workers=num_images) as executor:
            future_to_seed = {
                executor.submit(generate_single_image): s for s in range(num_images)
            }

            for future in as_completed(future_to_seed):
                try:
                    result = future.result()
                    if isinstance(result, Exception):
                        # 简化错误信息，不显示技术细节
                        errors.append(f"图像生成失败")
                    else:
                        pil_imgs, img_urls, errs = result
                        results_pil.extend(pil_imgs)
                        result_urls.extend(img_urls)
                        errors.extend(errs)
                except Exception as exc:
                    errors.append(f"图像生成异常")

        return results_pil, result_urls, errors

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A beautiful girl with long black hair, wearing a white dress, standing in a beautiful garden, looking at the camera.",
                    },
                ),
                "apikey": ("STRING", {"default": "请输入您的APIKEY: sk-xxxxxxx"}),
                "model": (
                    [
                        "gpt-image-2",
                    ],
                    {"default": "gpt-image-2"},
                ),
                "num_images": ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], {"default": 1}),
            },
            "optional": {
                "aspect_ratio": (
                    list(ASPECT_RATIO_STD_MAP.keys()),
                    {"default": "auto"},
                ),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")

    def _create_error_result(
        self, error_message: str, original_image: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        print(f"节点执行错误: {error_message}")
        if original_image is not None:
            image_out = original_image
        else:
            image_out = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        return {
            "ui": {"string": [error_message]},
            "result": (image_out, f"失败: {error_message}"),
        }

    def execute(self, **kwargs):
        prompt = kwargs.pop("prompt")
        model = kwargs.pop("model")
        apikey = kwargs.pop("apikey")
        aspect_ratio_label = kwargs.pop("aspect_ratio", None)
        aspect_ratio = _resolve_aspect_ratio(aspect_ratio_label, ASPECT_RATIO_STD_MAP)
        num_images = kwargs.pop("num_images", 1)

        # 收集可选输入图像
        images_in: List[torch.Tensor] = [
            kwargs.get(f"image_{i}")
            for i in range(1, 9)
            if kwargs.get(f"image_{i}") is not None
        ]
        for i in range(1, 9):
            kwargs.pop(f"image_{i}", None)

        image_data_urls: List[str] = []

        # 若提供了参考图，则将其转换为 base64 data URL
        if images_in:
            try:
                for image_tensor in images_in:
                    pil_images = tensor_to_pil(image_tensor)
                    if not pil_images:
                        continue

                    buffered = io.BytesIO()
                    pil_images[0].save(buffered, format="PNG")
                    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    image_data_urls.append(b64_str)

                if not image_data_urls:
                    return self._create_error_result(
                        "All input images could not be processed."
                    )
            except Exception as e:
                return self._create_error_result(
                    f"Image encoding failed: {format_error_message(e)}"
                )

        # 调用 GPT Image 接口
        try:
            with SuppressFalLogs():
                pil_images, image_urls, errors = self._execute_generation(
                    apikey=apikey,
                    final_prompt=prompt,
                    num_images=num_images,
                    model=model,
                    urls=image_data_urls,
                    aspect_ratio=aspect_ratio,
                )
        except Exception as e:
            return self._create_error_result(
                f"GPT Image API 调用失败: {format_error_message(e)}"
            )

        if not pil_images:
            error_msg = (
                "All image generations failed."
                if not images_in
                else "Image editing failed."
            )
            detail = f"; {errors}" if errors else ""
            return self._create_error_result(error_msg + detail)

        size_note = f" | aspectRatio: {aspect_ratio}" if aspect_ratio else ""
        failed_count = max(0, num_images - len(pil_images))
        fail_note = f" | 失败: {failed_count} 张" if failed_count > 0 else ""
        status = f"GPT Image | 模型: {model}{size_note} | 参考图片: {len(image_data_urls)} 张 | 成功生成: {len(pil_images)} 张{fail_note}"

        return {
            "ui": {"string": [status]},
            "result": (pil_to_tensor(pil_images), status),
        }


NODE_CLASS_MAPPINGS = {
    "Grsai_GPTImage": GrsaiGPTImage_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Grsai_GPTImage": "🎨 GrsAI GPT Image",
}
