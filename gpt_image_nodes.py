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
        create_generation_failed_image,
        pil_to_tensor,
        format_error_message,
        tensor_to_pil,
    )
except ImportError:
    from api_client import GrsaiAPI, GrsaiAPIError
    from config import default_config
    from utils import (
        create_generation_failed_image,
        pil_to_tensor,
        format_error_message,
        tensor_to_pil,
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
    ) -> Tuple[List[Any], List[str], List[str], List[str]]:
        task_results: Dict[
            int, Tuple[List[Any], List[str], List[str], List[str]]
        ] = {}

        def generate_single_image():
            api_client = None
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
                        "default": "A beautiful girl with long black hair, wearing a white dress, standing in a beautiful garden, looking at the camera.",
                    },
                ),
                "apikey": ("STRING", {"default": "请输入您的APIKEY: sk-xxxxxxx"}),
                "model": (
                    [
                        "gpt-image-2",
                        "gpt-image-2.5",
                    ],
                    {"default": "gpt-image-2"},
                ),
                "num_images": (
                    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    {"default": "1"},
                ),
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
        aspect_ratio_label = kwargs.pop("aspect_ratio", None)
        aspect_ratio = _resolve_aspect_ratio(aspect_ratio_label, ASPECT_RATIO_STD_MAP)
        num_images = int(kwargs.pop("num_images", "1"))

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
                pil_images, image_urls, errors, task_ids = self._execute_generation(
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
        success_count = min(num_images, len(image_urls))
        failed_count = max(0, num_images - success_count)
        fail_note = f" | 失败: {failed_count} 张" if failed_count > 0 else ""
        task_ids_text = "\n".join(task_ids)
        task_note = f" | 接口任务ID: {', '.join(task_ids)}" if task_ids else ""
        status = f"GPT Image | 模型: {model}{size_note} | 参考图片: {len(image_data_urls)} 张 | 成功生成: {success_count} 张{fail_note}{task_note}"

        return {
            "ui": {"string": [status]},
            "result": (pil_to_tensor(pil_images), status, task_ids_text),
        }


class GrsaiGPTImageVIP_Node:
    """
    GPT Image VIP 图像生成节点
    """

    FUNCTION = "execute"
    CATEGORY = "GrsAI/GPT Image 2 VIP"

    def _execute_generation(
        self,
        apikey: str,
        final_prompt: str,
        num_images: int,
        model: str,
        urls: list[str] = [],
        aspect_ratio: str = "auto",
        **kwargs,
    ) -> Tuple[List[Any], List[str], List[str], List[str]]:
        task_results: Dict[
            int, Tuple[List[Any], List[str], List[str], List[str]]
        ] = {}

        def generate_single_image():
            api_client = None
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
                        "default": "A beautiful girl with long black hair, wearing a white dress, standing in a beautiful garden, looking at the camera.",
                    },
                ),
                "apikey": ("STRING", {"default": "请输入您的APIKEY: sk-xxxxxxx"}),
                "model": (
                    [
                        "gpt-image-2-vip",
                    ],
                    {"default": "gpt-image-2-vip"},
                ),
                "num_images": (
                    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    {"default": "1"},
                ),
            },
            "optional": {
                "aspect_ratio": (
                    list(ASPECT_RATIO_VIP_MAP.keys()),
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
        aspect_ratio_label = kwargs.pop("aspect_ratio", None)
        aspect_ratio = _resolve_aspect_ratio(aspect_ratio_label, ASPECT_RATIO_VIP_MAP)
        num_images = int(kwargs.pop("num_images", "1"))

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
                pil_images, image_urls, errors, task_ids = self._execute_generation(
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
        success_count = min(num_images, len(image_urls))
        failed_count = max(0, num_images - success_count)
        fail_note = f" | 失败: {failed_count} 张" if failed_count > 0 else ""
        task_ids_text = "\n".join(task_ids)
        task_note = f" | 接口任务ID: {', '.join(task_ids)}" if task_ids else ""
        status = f"GPT Image | 模型: {model}{size_note} | 参考图片: {len(image_data_urls)} 张 | 成功生成: {success_count} 张{fail_note}{task_note}"

        return {
            "ui": {"string": [status]},
            "result": (pil_to_tensor(pil_images), status, task_ids_text),
        }


class GrsaiGPTImage25_Node(GrsaiGPTImageVIP_Node):
    """
    GPT Image 2.5 Flare / Sunburst 图像生成节点。

    两个模型与 gpt-image-2-vip 使用相同的尺寸、批量和参考图参数。
    """

    CATEGORY = "GrsAI/GPT Image 2.5"

    @classmethod
    def INPUT_TYPES(cls):
        input_types = super().INPUT_TYPES()
        input_types["required"]["model"] = (
            [
                "gpt-image-2.5-flare",
                "gpt-image-2.5-sunburst",
            ],
            {"default": "gpt-image-2.5-flare"},
        )
        return input_types


NODE_CLASS_MAPPINGS = {
    "Grsai_GPTImage": GrsaiGPTImage_Node,
    "Grsai_GPTImageVIP": GrsaiGPTImageVIP_Node,
    "Grsai_GPTImage25": GrsaiGPTImage25_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Grsai_GPTImage": "🎨 GrsAI GPT Image",
    "Grsai_GPTImageVIP": "🎨 GrsAI GPT Image 2 VIP",
    "Grsai_GPTImage25": "🎨 GrsAI GPT Image 2.5 Flare / Sunburst",
}
