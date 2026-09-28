"""
ComfyUI节点实现
定义 Nano Banana 图像生成节点（文生图 / 图生图 / 多图）
"""

import logging
from typing import Any, Tuple, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import torch

# 尝试相对导入，如果失败则使用绝对导入
try:
    from .api_client import GrsaiAPI, GrsaiAPIError
    from .config import default_config
    from .utils import (
        create_generation_failed_image,
        pil_to_tensor,
        format_error_message,
        tensor_to_base64,
    )
except ImportError:
    from api_client import GrsaiAPI, GrsaiAPIError
    from config import default_config
    from utils import (
        create_generation_failed_image,
        pil_to_tensor,
        format_error_message,
        tensor_to_base64,
    )


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


class GrsaiNanoBanana2_Node:
    """
    Nano Banana 图像生成节点
    - 可选多图作为参考：不输入图像时为文生图；输入1张或多张时为图生图
    """

    FUNCTION = "execute"
    CATEGORY = "GrsAI/Nano Banana 2"

    def _execute_generation(
        self,
        grsai_api_key: str,
        final_prompt: str,
        num_images: int,
        model: str,
        urls: list[str] = [],
        aspect_ratio: str = "auto",
        image_size: str = "1K",
        **kwargs,
    ) -> Tuple[List[Any], List[str], List[str], List[str]]:
        task_results: Dict[
            int, Tuple[List[Any], List[str], List[str], List[str]]
        ] = {}

        def generate_single_image():
            api_client = None
            try:
                api_client = GrsaiAPI(api_key=grsai_api_key)
                api_params = {
                    "prompt": final_prompt,
                    "model": model,
                    "urls": urls,
                    "aspect_ratio": aspect_ratio,
                    "image_size": image_size,
                }
                api_params.update(kwargs)
                pil_imgs, img_urls, errs = api_client.banana_generate_image(
                    **api_params
                )
                return pil_imgs, img_urls, errs, api_client.last_task_id
            except Exception as e:
                if api_client is not None and api_client.last_task_id:
                    setattr(e, "task_id", api_client.last_task_id)
                return e

        with ThreadPoolExecutor(max_workers=num_images) as executor:
            future_to_index = {
                executor.submit(generate_single_image): index
                for index in range(num_images)
            }

            for future in as_completed(future_to_index):
                task_index = future_to_index[future]
                try:
                    result = future.result()
                    if isinstance(result, Exception):
                        task_id = getattr(result, "task_id", None)
                        display_task_id = task_id or "未创建（请求未提交）"
                        task_note = f" [接口任务ID: {display_task_id}]"
                        error = (
                            f"批量任务 {task_index + 1} 生成失败: "
                            f"{result}{task_note}"
                        )
                        print(f"❌ {error}")
                        task_results[task_index] = (
                            [create_generation_failed_image(error_message=error)],
                            [],
                            [error],
                            [task_id or "未创建"],
                        )
                    else:
                        pil_imgs, img_urls, errs, task_id = result
                        task_ids = [task_id or "未创建"]
                        if pil_imgs:
                            task_results[task_index] = (
                                pil_imgs,
                                img_urls,
                                errs,
                                task_ids,
                            )
                        else:
                            error = "; ".join(errs) or "未返回可用图片"
                            print(f"❌ 批量任务 {task_index + 1} 生成失败: {error}")
                            task_results[task_index] = (
                                [create_generation_failed_image(error_message=error)],
                                [],
                                [error],
                                task_ids,
                            )
                except Exception as exc:
                    task_id = getattr(exc, "task_id", None)
                    display_task_id = task_id or "未创建（请求未提交）"
                    error = (
                        f"批量任务 {task_index + 1} 生成异常: {exc} "
                        f"[接口任务ID: {display_task_id}]"
                    )
                    print(f"❌ {error}")
                    task_results[task_index] = (
                        [create_generation_failed_image(error_message=error)],
                        [],
                        [error],
                        [task_id or "未创建"],
                    )

        results_pil, result_urls, errors, task_ids = [], [], [], []
        for task_index in range(num_images):
            pil_imgs, img_urls, task_errors, current_task_ids = task_results[task_index]
            results_pil.extend(pil_imgs)
            result_urls.extend(img_urls)
            errors.extend(task_errors)
            task_ids.extend(current_task_ids)
        return results_pil, result_urls, errors, task_ids

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
                "apikey": ("STRING", {"default": "请输入您的APIKEY: sk-xxxxxxx"}),
                "model": (
                    [
                        "nano-banana-2",
                        "nano-banana-2-cl",
                        "nano-banana-2-2k-cl",
                        "nano-banana-2-4k-cl",
                    ],
                    {"default": "nano-banana-2"},
                ),
                "num_images": (
                    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    {"default": "1"},
                ),
            },
            "optional": {
                "aspect_ratio": (
                    [
                        "auto",
                        "1:1",
                        "16:9",
                        "9:16",
                        "4:3",
                        "3:4",
                        "3:2",
                        "2:3",
                        "5:4",
                        "4:5",
                        "21:9",
                        "4:1",
                        "1:4",
                        "8:1",
                        "1:8",
                    ],
                    {"default": "auto"},
                ),
                "image_size": (
                    [
                        "1K",
                        "2K",
                        "4K",
                    ],
                    {"default": "1K"},
                ),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "status", "api_task_ids")

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")

    def _create_error_result(
        self, error_message: str, original_image: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        full_error_message = (
            f"{error_message}\n接口任务ID: 未创建（请求未成功提交）"
        )
        print(f"节点执行错误: {full_error_message}")
        if original_image is not None:
            height, width = original_image.shape[1:3]
        else:
            width = height = 1024
        image_out = pil_to_tensor(
            create_generation_failed_image(
                width=width, height=height, error_message=full_error_message
            )
        )

        return {
            "ui": {"string": [full_error_message]},
            "result": (image_out, f"失败: {full_error_message}", "未创建"),
        }

    def execute(self, **kwargs):
        prompt = kwargs.pop("prompt")
        model = kwargs.pop("model")
        apikey = kwargs.pop("apikey")
        aspect_ratio = kwargs.pop("aspect_ratio", None)
        image_size = kwargs.pop("image_size", "1K")
        num_images = int(kwargs.pop("num_images", "1"))

        # 收集可选输入图像
        images_in: List[torch.Tensor] = [
            kwargs.get(f"image_{i}")
            for i in range(1, 11)
            if kwargs.get(f"image_{i}") is not None
        ]
        for i in range(1, 11):
            kwargs.pop(f"image_{i}", None)

        # 若提供了参考图，则转换为 base64 data URI 直接传入 urls 参数
        image_data_uris: List[str] = []
        if images_in:
            try:
                for image_tensor in images_in:
                    try:
                        base64_str = tensor_to_base64(image_tensor, image_format="png")
                    except ValueError:
                        continue
                    image_data_uris.append(base64_str)

                if not image_data_uris:
                    return self._create_error_result(
                        "All input images could not be processed."
                    )
            except Exception as e:
                return self._create_error_result(
                    f"Image processing failed: {format_error_message(e)}"
                )

        # 调用 Nano Banana 接口
        try:
            with SuppressFalLogs():
                pil_images, image_urls, errors, task_ids = self._execute_generation(
                    grsai_api_key=apikey,
                    final_prompt=prompt,
                    num_images=num_images,
                    model=model,
                    urls=image_data_uris,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
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

        size_note = f" | imageSize: {image_size}" if image_size else ""
        success_count = min(num_images, len(image_urls))
        failed_count = max(0, num_images - success_count)
        fail_note = f" | 失败: {failed_count} 张" if failed_count > 0 else ""
        task_ids_text = "\n".join(task_ids)
        task_note = f" | 接口任务ID: {', '.join(task_ids)}" if task_ids else ""
        status = f"Nano Banana | 模型: {model}{size_note} | 参考图片: {len(image_data_uris)} 张 | 成功生成: {success_count} 张{fail_note}{task_note}"

        return {
            "ui": {"string": [status]},
            "result": (pil_to_tensor(pil_images), status, task_ids_text),
        }


NODE_CLASS_MAPPINGS = {
    "Grsai_NanoBanana2": GrsaiNanoBanana2_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Grsai_NanoBanana2": "🍌 GrsAI Nano Banana 2 - Text/Image",
}
