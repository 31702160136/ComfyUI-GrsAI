"""
ComfyUI节点实现
定义 Nano Banana 图像生成节点（文生图 / 图生图 / 多图）
"""

import os
import tempfile
import logging
from typing import Any, Tuple, Optional, Dict, List

import torch

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .upload import upload_file_zh
    from .api_client import GrsaiAPI, GrsaiAPIError
    from .config import default_config
    from .utils import (
        pil_to_tensor,
        format_error_message,
        tensor_to_pil,
    )
except ImportError:
    from upload import upload_file_zh
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


class GrsaiNanoBanana_Node:
    """
    Nano Banana 图像生成节点
    - 可选多图作为参考：不输入图像时为文生图；输入1张或多张时为图生图
    """

    FUNCTION = "execute"
    CATEGORY = "GrsAI/Nano Banana"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Create a high-quality studio shot of a ripe banana on a matte surface, soft shadows, natural lighting.",
                    },
                ),
                "model": (
                    ["nano-banana", "nano-banana-fast"],
                    {"default": "nano-banana-fast"},
                ),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
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
        grsai_api_key = default_config.get_api_key()
        if not grsai_api_key:
            return self._create_error_result(default_config.api_key_error_message)

        prompt = kwargs.pop("prompt")
        model = kwargs.pop("model")

        # 收集可选输入图像
        images_in: List[torch.Tensor] = [
            kwargs.get(f"image_{i}")
            for i in range(1, 7)
            if kwargs.get(f"image_{i}") is not None
        ]
        for i in range(1, 7):
            kwargs.pop(f"image_{i}", None)

        uploaded_urls: List[str] = []
        temp_files: List[str] = []

        # 若提供了参考图，则上传获取URL
        if images_in:
            try:
                for i, image_tensor in enumerate(images_in):
                    pil_images = tensor_to_pil(image_tensor)
                    if not pil_images:
                        continue

                    with tempfile.NamedTemporaryFile(
                        suffix=f"_{i}.png", delete=False
                    ) as temp_file:
                        pil_images[0].save(temp_file, "PNG")
                        temp_files.append(temp_file.name)

                    with SuppressFalLogs():
                        uploaded_urls.append(
                            upload_file_zh(
                                api_key=grsai_api_key, file_path=temp_files[-1]
                            )
                        )

                if not uploaded_urls:
                    return self._create_error_result(
                        "All input images could not be processed or uploaded."
                    )
            except Exception as e:
                return self._create_error_result(
                    f"Image upload failed: {format_error_message(e)}"
                )
            finally:
                for path in temp_files:
                    if os.path.exists(path):
                        os.unlink(path)

        # 调用 Nano Banana 接口
        try:
            api_client = GrsaiAPI(api_key=grsai_api_key)
            with SuppressFalLogs():
                pil_images, image_urls, errors = api_client.banana_generate_image(
                    prompt=prompt, model=model, urls=uploaded_urls
                )
        except Exception as e:
            return self._create_error_result(
                f"Nano Banana API 调用失败: {format_error_message(e)}"
            )

        if not pil_images:
            error_msg = (
                "All image generations failed."
                if not images_in
                else "Image editing failed."
            )
            detail = f"; {errors}" if errors else ""
            return self._create_error_result(error_msg + detail)

        status = f"Nano Banana | 参考图片: {len(uploaded_urls)} 张 | 成功生成: {len(pil_images)} 张"

        return {
            "ui": {"string": [status]},
            "result": (pil_to_tensor(pil_images), status),
        }


NODE_CLASS_MAPPINGS = {
    "Grsai_NanoBanana": GrsaiNanoBanana_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Grsai_NanoBanana": "🍌 GrsAI Nano Banana - Text/Image",
}
