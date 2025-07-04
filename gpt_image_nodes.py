"""
ComfyUI节点实现
定义GPT Image图像生成节点
"""

import torch
import os
import tempfile
import logging
from typing import Any, Tuple, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .upload import upload_file_zh
    from .api_client import GrsaiAPI, GrsaiAPIError
    from .config import default_config
    from .utils import (
        download_image,
        pil_to_tensor,
        format_error_message,
        tensor_to_pil,
    )
except ImportError:
    from upload import upload_file_zh
    from api_client import GrsaiAPI, GrsaiAPIError
    from config import default_config
    from utils import download_image, pil_to_tensor, format_error_message, tensor_to_pil


class SuppressFalLogs:
    """临时抑制FAL相关的详细HTTP日志的上下文管理器"""

    def __init__(self):
        self.loggers_to_suppress = [
            "httpx",
            "httpcore",
            "urllib3.connectionpool",
        ]
        self.original_levels = {}

    def __enter__(self):
        # 保存原始日志级别并设置为WARNING以上
        for logger_name in self.loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            self.original_levels[logger_name] = logger.level
            logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始日志级别
        for logger_name, original_level in self.original_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(original_level)


class _GPTImageNodeBase:
    """
    所有GPT Image节点的内部基类，处理通用逻辑。
    """

    FUNCTION = "execute"
    CATEGORY = "GrsAI/GPT Image"

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

    def _execute_generation(
        self,
        grsai_api_key: str,
        final_prompt: str,
        model: str,
        urls: list[str],
        variants: int,
        **kwargs,
    ) -> Tuple[List[Any], List[str], List[str]]:
        results_pil, result_urls, errors = [], [], []

        def generate_image():
            try:
                api_client = GrsaiAPI(api_key=grsai_api_key)
                api_params = {
                    "prompt": final_prompt,
                    "model": model,
                    "urls": urls,
                    "variants": variants,
                }
                api_params.update(kwargs)
                pil_images, image_urls, api_errors = (
                    api_client.gpt_image_generate_image(**api_params)
                )
                return pil_images, image_urls, api_errors
            except Exception as e:
                return None, None, e

        result = generate_image()
        pil_images, image_urls, api_errors = result
        if pil_images is not None:
            results_pil.extend(pil_images)
        if image_urls is not None:
            result_urls.extend(image_urls)
        if isinstance(api_errors, list):
            errors.extend(api_errors)
        elif isinstance(api_errors, Exception):
            errors.append(str(api_errors))

        return results_pil, result_urls, errors


# 节点1: 文生图
class GPTImage_TextToImage(_GPTImageNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A colorful and stylized mechanical bird sculpture, with bright blue and green body, orange accent stripes, and a white head. The bird has a smooth, polished surface and is positioned as if perched on a branch. The sculpture's pieces are segmented, giving it a modular, toy-like appearance, with visible joints between the segments. The background is a soft, blurred green to evoke a natural, outdoors feel. The word 'FLUX' is drawn with a large white touch on it, with distinct textures",
                    },
                ),
                "variants": ([1, 2, 3, 4], {"default": 1}),
                "size": (
                    ["auto", "1:1", "2:3", "3:2"],
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")

    def execute(self, **kwargs):
        grsai_api_key = default_config.get_api_key()
        if not grsai_api_key:
            return self._create_error_result(default_config.api_key_error_message)

        variants = kwargs.pop("variants")
        final_prompt = kwargs.pop("prompt")
        model = "sora-image"

        results_pil, result_urls, errors = self._execute_generation(
            grsai_api_key, final_prompt, model, [], variants, **kwargs
        )

        if not results_pil:
            return self._create_error_result(
                f"All image generations failed.\n{'; '.join(errors)}"
            )

        success_count = len(results_pil)
        final_status = f"文生图模式 | 成功生成: {success_count}/{variants} 张图像"
        if errors:
            final_status += f" | 失败: {len(errors)} 张"

        return {
            "ui": {"string": [final_status]},
            "result": (pil_to_tensor(results_pil), final_status),
        }


# 节点3: 图生图
class GPTImage_ImageToImage(_GPTImageNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "The character is sitting cross-legged on the sofa, and the Dalmatian is lying on the blanket sleeping.",
                    },
                ),
                "variants": ([1, 2, 3, 4], {"default": 1}),
                "size": (
                    ["auto", "1:1", "2:3", "3:2"],
                    {"default": "auto"},
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

    def execute(self, **kwargs):
        images_in = [
            kwargs.get(f"image_{i}")
            for i in range(1, 5)
            if kwargs.get(f"image_{i}") is not None
        ]

        if not images_in:
            return self._create_error_result(
                "Error: node requires at least one image input."
            )

        grsai_api_key = default_config.get_api_key()
        if not grsai_api_key:
            return self._create_error_result(default_config.api_key_error_message)

        os.environ["GRSAI_KEY"] = grsai_api_key

        uploaded_urls = []
        temp_files = []
        try:
            for i, image_tensor in enumerate(images_in):
                if image_tensor is None:
                    continue

                pil_images = tensor_to_pil(image_tensor)
                if not pil_images:
                    continue

                with tempfile.NamedTemporaryFile(
                    suffix=f"_{i}.png", delete=False
                ) as temp_file:
                    pil_images[0].save(temp_file, "PNG")
                    temp_files.append(temp_file.name)

                with SuppressFalLogs():
                    uploaded_urls.append(upload_file_zh(temp_files[-1]))

            if not uploaded_urls:
                return self._create_error_result(
                    "All input images could not be processed or uploaded."
                )

            final_prompt = kwargs["prompt"]

        except Exception as e:
            return self._create_error_result(
                f"Multi-Image upload failed: {format_error_message(e)}"
            )
        finally:
            for path in temp_files:
                if os.path.exists(path):
                    os.unlink(path)

        variants = kwargs.pop("variants")
        model = "sora-image"
        kwargs.pop("prompt")
        for i in range(1, 5):
            kwargs.pop(f"image_{i}", None)

        results_pil, result_urls, errors = self._execute_generation(
            grsai_api_key,
            final_prompt,
            model,
            uploaded_urls,
            variants,
            **kwargs,
        )

        if not results_pil:
            return self._create_error_result(
                f"All image generations failed.\n{'; '.join(errors)}"
            )

        success_count = len(results_pil)
        final_status = f"图生图模式 | 参考图片: {len(uploaded_urls)} 张 | 成功生成: {success_count}/{variants} 张图像"
        if errors:
            final_status += f" | 失败: {len(errors)} 张"

        return {
            "ui": {"string": [final_status]},
            "result": (pil_to_tensor(results_pil), final_status),
        }


NODE_CLASS_MAPPINGS = {
    "GPTImage_TextToImage": GPTImage_TextToImage,
    "GPTImage_ImageToImage": GPTImage_ImageToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImage_TextToImage": "🎨 GrsAI GPT Image - Text to Image",
    "GPTImage_ImageToImage": "🎨 GrsAI GPT Image - Image to Image",
}
