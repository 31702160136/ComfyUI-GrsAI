"""
ComfyUI节点实现
定义 GPT Image 图像生成节点（文生图 / 图生图 / 多图）
"""

import os
import tempfile
import logging
from typing import Any, Tuple, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

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
                    "size": aspect_ratio,
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
                    [
                        "auto",
                        "auto",
                        "1:1",
                        "3:2",
                        "2:3",
                        "16:9",
                        "9:16",
                        "5:4",
                        "4:5",
                        "4:3",
                        "3:4",
                        "21:9",
                        "9:21",
                        "1:3",
                        "3:1",
                        "2:1",
                        "1:2",
                    ],
                    {"default": "auto"},
                ),
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
        prompt = kwargs.pop("prompt")
        model = kwargs.pop("model")
        apikey = kwargs.pop("apikey")
        aspect_ratio = kwargs.pop("aspect_ratio", None)
        num_images = kwargs.pop("num_images", 1)

        # 收集可选输入图像
        images_in: List[torch.Tensor] = [
            kwargs.get(f"image_{i}")
            for i in range(1, 11)
            if kwargs.get(f"image_{i}") is not None
        ]
        for i in range(1, 11):
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
                            upload_file_zh(api_key=apikey, file_path=temp_files[-1])
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

        # 调用 GPT Image 接口
        try:
            with SuppressFalLogs():
                pil_images, image_urls, errors = self._execute_generation(
                    apikey=apikey,
                    final_prompt=prompt,
                    num_images=num_images,
                    model=model,
                    urls=uploaded_urls,
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
        status = f"GPT Image | 模型: {model}{size_note} | 参考图片: {len(uploaded_urls)} 张 | 成功生成: {len(pil_images)} 张"

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
