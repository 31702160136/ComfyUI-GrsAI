"""
ComfyUI节点实现
定义Flux-Kontext图像生成节点
"""

import torch
import random
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


class _GrsaiFluxKontextNodeBase:
    """
    所有Flux-Kontext节点的内部基类，处理通用逻辑。
    """

    FUNCTION = "execute"
    CATEGORY = "GrsAI/Flux.1"

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
        num_images: int,
        seed: int,
        model: str,
        urls: list[str] = [],
        **kwargs,
    ) -> Tuple[List[Any], List[str], List[str]]:
        results_pil, result_urls, errors = [], [], []

        def generate_single_image(current_seed):
            try:
                api_client = GrsaiAPI(api_key=grsai_api_key)
                api_params = {
                    "prompt": final_prompt,
                    "model": model,
                    "seed": current_seed,
                    "urls": urls,
                }
                api_params.update(kwargs)
                pil_image, url = api_client.flux_generate_image(**api_params)
                return pil_image, url
            except Exception as e:
                return e

        with ThreadPoolExecutor(max_workers=min(num_images, 4)) as executor:
            # 限制seed在32位整数范围内，避免API解析错误
            seeds = [
                seed + i if seed != 0 else random.randint(1, 2147483647)
                for i in range(num_images)
            ]
            future_to_seed = {
                executor.submit(generate_single_image, s): s for s in seeds
            }

            for future in as_completed(future_to_seed):
                try:
                    result = future.result()
                    if isinstance(result, Exception):
                        # 简化错误信息，不显示技术细节
                        errors.append(f"图像生成失败")
                    else:
                        pil_img, url = result
                        results_pil.append(pil_img)
                        result_urls.append(url)
                except Exception as exc:
                    errors.append(f"图像生成异常")

        return results_pil, result_urls, errors


# 节点1: 文生图
class GrsaiFluxKontext_TextToImage(_GrsaiFluxKontextNodeBase):
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
                "model": (
                    [
                        "flux-pro-1.1",
                        "flux-pro-1.1-ultra",
                        "flux-kontext-pro",
                        "flux-kontext-max",
                    ],
                    {"default": "flux-kontext-max"},
                ),
                "num_images": ([1, 2, 3, 4], {"default": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                # "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "aspect_ratio": (
                    default_config.SUPPORTED_ASPECT_RATIOS,
                    {"default": "1:1"},
                ),
                # "output_format": (default_config.SUPPORTED_OUTPUT_FORMATS, {"default": "png"}),
                "safety_tolerance": ("INT", {"default": 6, "min": 0, "max": 6}),
                "prompt_upsampling": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")

    def execute(self, **kwargs):
        grsai_api_key = default_config.get_api_key()
        if not grsai_api_key:
            return self._create_error_result(default_config.api_key_error_message)

        num_images = kwargs.pop("num_images")
        seed = kwargs.pop("seed")
        final_prompt = kwargs.pop("prompt")
        model = kwargs.pop("model")

        results_pil, result_urls, errors = self._execute_generation(
            grsai_api_key, final_prompt, num_images, seed, model, **kwargs
        )

        if not results_pil:
            return self._create_error_result(
                f"All image generations failed.\n{'; '.join(errors)}"
            )

        success_count = len(results_pil)
        final_status = f"文生图模式 | 成功生成: {success_count}/{num_images} 张图像"
        if errors:
            final_status += f" | 失败: {len(errors)} 张"

        return {
            "ui": {"string": [final_status]},
            "result": (pil_to_tensor(results_pil), final_status),
        }


# 节点2: 图生图 (单图)
class GrsaiFluxKontext_ImageToImage(_GrsaiFluxKontextNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "The character is sitting cross-legged on the sofa, and the Dalmatian is lying on the blanket sleeping.",
                    },
                ),
                "model": (
                    ["flux-kontext-pro", "flux-kontext-max"],
                    {"default": "flux-kontext-max"},
                ),
                "num_images": ([1, 2, 3, 4], {"default": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                # "guidance_scale": (
                #     "FLOAT",
                #     {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1},
                # ),
                # "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "aspect_ratio": (
                    default_config.SUPPORTED_ASPECT_RATIOS,
                    {"default": "1:1"},
                ),
                # "output_format": (
                #     default_config.SUPPORTED_OUTPUT_FORMATS,
                #     {"default": "png"},
                # ),
                "safety_tolerance": ("INT", {"default": 6, "min": 0, "max": 6}),
                "prompt_upsampling": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")

    def execute(self, image: torch.Tensor, **kwargs):
        grsai_api_key = default_config.get_api_key()
        if not grsai_api_key:
            return self._create_error_result(
                default_config.api_key_error_message, image
            )

        grsai_key = default_config.get_api_key()
        if not grsai_key:
            return self._create_error_result(
                default_config.api_key_error_message, image
            )

        os.environ["GRSAI_KEY"] = grsai_key
        temp_file_path = None
        uploaded_url = ""
        try:
            pil_images = tensor_to_pil(image)
            if not pil_images:
                return self._create_error_result("Cannot convert input image.", image)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                pil_images[0].save(temp_file, "PNG")
                temp_file_path = temp_file.name

            with SuppressFalLogs():
                uploaded_url = upload_file_zh(temp_file_path)
            final_prompt = f"{kwargs['prompt']}"

        except Exception as e:
            return self._create_error_result(
                f"Image-to-Image preparation failed: {format_error_message(e)}", image
            )
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        num_images = kwargs.pop("num_images")
        seed = kwargs.pop("seed")
        model = kwargs.pop("model")
        kwargs.pop("prompt")

        results_pil, result_urls, errors = self._execute_generation(
            grsai_api_key,
            final_prompt,
            num_images,
            seed,
            model,
            [uploaded_url],
            **kwargs,
        )

        if not results_pil:
            return self._create_error_result(
                f"All image generations failed.\n{'; '.join(errors)}", image
            )

        success_count = len(results_pil)
        final_status = f"图生图模式 | 成功生成: {success_count}/{num_images} 张图像"
        if errors:
            final_status += f" | 失败: {len(errors)} 张"

        return {
            "ui": {"string": [final_status]},
            "result": (pil_to_tensor(results_pil), final_status),
        }


# 节点3: 多图生图
class GrsaiFluxKontext_MultiImageToImage(_GrsaiFluxKontextNodeBase):
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
                "model": (
                    ["flux-kontext-pro", "flux-kontext-max"],
                    {"default": "flux-kontext-max"},
                ),
                "num_images": ([1, 2, 3, 4], {"default": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                # "guidance_scale": (
                #     "FLOAT",
                #     {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1},
                # ),
                # "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "aspect_ratio": (
                    default_config.SUPPORTED_ASPECT_RATIOS,
                    {"default": "1:1"},
                ),
                # "output_format": (
                #     default_config.SUPPORTED_OUTPUT_FORMATS,
                #     {"default": "png"},
                # ),
                "safety_tolerance": ("INT", {"default": 6, "min": 0, "max": 6}),
                "prompt_upsampling": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
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
                "Error: Multi-Image node requires at least one image input."
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

        num_images = kwargs.pop("num_images")
        seed = kwargs.pop("seed")
        model = kwargs.pop("model")
        kwargs.pop("prompt")
        for i in range(1, 5):
            kwargs.pop(f"image_{i}", None)

        results_pil, result_urls, errors = self._execute_generation(
            grsai_api_key,
            final_prompt,
            num_images,
            seed,
            model,
            uploaded_urls,
            **kwargs,
        )

        if not results_pil:
            return self._create_error_result(
                f"All image generations failed.\n{'; '.join(errors)}"
            )

        success_count = len(results_pil)
        final_status = f"多图生图模式 | 参考图片: {len(uploaded_urls)} 张 | 成功生成: {success_count}/{num_images} 张图像"
        if errors:
            final_status += f" | 失败: {len(errors)} 张"

        return {
            "ui": {"string": [final_status]},
            "result": (pil_to_tensor(results_pil), final_status),
        }


NODE_CLASS_MAPPINGS = {
    "GrsaiFluxKontext_TextToImage": GrsaiFluxKontext_TextToImage,
    "GrsaiFluxKontext_ImageToImage": GrsaiFluxKontext_ImageToImage,
    "GrsaiFluxKontext_MultiImageToImage": GrsaiFluxKontext_MultiImageToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrsaiFluxKontext_TextToImage": "🎨 GrsAI Flux.1 Kontext - Text to Image",
    "GrsaiFluxKontext_ImageToImage": "🎨 GrsAI Flux.1 Kontext - Editing",
    "GrsaiFluxKontext_MultiImageToImage": "🎨 GrsAI Flux.1 Kontext - Editing (Multi Image)",
}
