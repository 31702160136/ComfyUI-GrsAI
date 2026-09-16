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


# gpt-image-2.5 尺寸映射：比例 -> 图像等级(1K/2K/4K) -> 实际发送给 API 的尺寸
GPT_IMAGE_ASPECT_RATIO_MAP: Dict[str, Dict[str, str]] = {
    # 1:1
    "1:1": {
        "1K": "1024x1024",
        "2K": "2048x2048",
        "4K": "2880x2880",
    },
    # 16:9
    "16:9": {
        "1K": "1280x720",
        "2K": "2048x1152",
        "4K": "3840x2160",
    },
    # 9:16
    "9:16": {
        "1K": "720x1280",
        "2K": "1152x2048",
        "4K": "2160x3840",
    },
    # 4:3
    "4:3": {
        "1K": "1152x864",
        "2K": "2304x1728",
        "4K": "3264x2448",
    },
    # 3:4
    "3:4": {
        "1K": "864x1152",
        "2K": "1728x2304",
        "4K": "2448x3264",
    },
    # 3:2
    "3:2": {
        "1K": "1536x1024",
        "2K": "2048x1360",
        "4K": "3504x2336",
    },
    # 2:3
    "2:3": {
        "1K": "1024x1536",
        "2K": "1360x2048",
        "4K": "2336x3504",
    },
    # 5:4
    "5:4": {
        "1K": "1120x896",
        "2K": "2240x1792",
        "4K": "3200x2560",
    },
    # 4:5
    "4:5": {
        "1K": "896x1120",
        "2K": "1792x2240",
        "4K": "2560x3200",
    },
    # 21:9
    "21:9": {
        "1K": "1456x624",
        "2K": "2912x1248",
        "4K": "3840x1648",
    },
    # 9:21
    "9:21": {
        "1K": "624x1456",
        "2K": "1248x2912",
        "4K": "1648x3840",
    },
    # 1:3
    "1:3": {
        "2K": "688x2048",
        "4K": "1280x3840",
    },
    # 3:1
    "3:1": {
        "2K": "2048x688",
        "4K": "3840x1280",
    },
    # 2:1
    "2:1": {
        "1K": "1536x768",
        "2K": "3072x1536",
        "4K": "3840x1920",
    },
    # 1:2
    "1:2": {
        "1K": "768x1536",
        "2K": "1536x3072",
        "4K": "1920x3840",
    },
}

# gpt-image-2 / gpt-image-2.5 支持的比例（仅 1K 像素值）
GPT_IMAGE_NON_VIP_RATIOS: Dict[str, str] = {
    "1:1": "1024x1024",
    "16:9": "1672x941",
    "9:16": "941x1672",
    "4:3": "1443x1090",
    "3:4": "1090x1443",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
    "5:4": "1408x1120",
    "4:5": "1120x1408",
    "21:9": "1920x832",
    "9:21": "832x1920",
    "1:2": "896x1792",
    "2:1": "1792x896",
}

# gpt-image-2 / gpt-image-2.5 节点的比例下拉选项：比例（长*高）
GPT_IMAGE_NON_VIP_ASPECT_RATIO_OPTIONS: List[str] = ["auto"] + [
    f"{ratio}（{pixels}）" for ratio, pixels in GPT_IMAGE_NON_VIP_RATIOS.items()
]

# vip 模型的比例下拉选项
GPT_IMAGE_VIP_ASPECT_RATIO_OPTIONS: List[str] = ["auto"] + list(
    GPT_IMAGE_ASPECT_RATIO_MAP.keys()
)

GPT_IMAGE_SIZE_OPTIONS: List[str] = ["1K", "2K", "4K"]


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


class GrsaiGPTImage_Node:
    """
    GPT Image 图像生成节点（gpt-image-2 / gpt-image-2.5）
    """

    FUNCTION = "execute"
    CATEGORY = "GrsAI/GPT Image"

    MODELS: List[str] = ["gpt-image-2", "gpt-image-2.5"]
    DEFAULT_MODEL: str = "gpt-image-2"
    HAS_ASPECT_RATIO: bool = True
    HAS_IMAGE_SIZE: bool = False
    VIP: bool = False
    ASPECT_RATIO_OPTIONS: List[str] = GPT_IMAGE_NON_VIP_ASPECT_RATIO_OPTIONS
    DEFAULT_ASPECT_RATIO: str = "auto"
    # 无 aspect_ratio 选项的节点，按该比例取对应 image_size 的像素值
    NO_RATIO_DEFAULT: str = "1:1"
    BACKGROUND_OPTIONS: Optional[List[str]] = ["否", "是（2.5不支持，别选）"]
    DEFAULT_BACKGROUND: str = "否"
    QUALITY_OPTIONS: Optional[List[str]] = None
    QUALITY_VALUE_MAP: Dict[str, str] = {}
    DEFAULT_QUALITY: str = "auto"

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
        inputs = {
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
                    cls.MODELS,
                    {"default": cls.DEFAULT_MODEL},
                ),
                "num_images": (
                    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    {"default": "1"},
                ),
            },
            "optional": {
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
        if cls.HAS_ASPECT_RATIO:
            inputs["optional"]["aspect_ratio"] = (
                cls.ASPECT_RATIO_OPTIONS,
                {"default": cls.DEFAULT_ASPECT_RATIO},
            )
        if cls.HAS_IMAGE_SIZE:
            inputs["optional"]["image_size"] = (
                GPT_IMAGE_SIZE_OPTIONS,
                {"default": "1K"},
            )
        if cls.BACKGROUND_OPTIONS:
            inputs["optional"]["透明背景"] = (
                cls.BACKGROUND_OPTIONS,
                {"default": cls.DEFAULT_BACKGROUND},
            )
        if cls.QUALITY_OPTIONS:
            inputs["optional"]["quality"] = (
                cls.QUALITY_OPTIONS,
                {"default": cls.DEFAULT_QUALITY},
            )
        return inputs

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

    def _resolve_aspect_ratio(
        self, label: Optional[str], image_size: str
    ) -> Optional[str]:
        """将比例下拉值转换为实际发送给 API 的尺寸值。"""
        if not self.HAS_ASPECT_RATIO:
            sizes = GPT_IMAGE_ASPECT_RATIO_MAP.get(self.NO_RATIO_DEFAULT)
            if sizes:
                return sizes.get(image_size) or next(iter(sizes.values()))
            return "auto"
        if label is None or label == "auto":
            return "auto"
        # 兼容旧工作流中已保存的 “1024x1024 (1:1)” 标签
        if label in ASPECT_RATIO_STD_MAP:
            return ASPECT_RATIO_STD_MAP[label]
        if not self.VIP:
            # 下拉格式：比例（长*高），发送括号内的 1K 像素值
            if "（" in label:
                return label.split("（", 1)[1].rstrip("）").strip()
            return label
        sizes = GPT_IMAGE_ASPECT_RATIO_MAP.get(label)
        if sizes:
            return sizes.get(image_size) or next(iter(sizes.values()))
        return label

    def execute(self, **kwargs):
        prompt = kwargs.pop("prompt")
        model = kwargs.pop("model")
        apikey = kwargs.pop("apikey")
        aspect_ratio_label = kwargs.pop("aspect_ratio", "auto")
        image_size = kwargs.pop("image_size", "1K")
        background = kwargs.pop("透明背景", "否")
        quality = kwargs.pop("quality", None)
        aspect_ratio = self._resolve_aspect_ratio(aspect_ratio_label, image_size)
        num_images = int(kwargs.pop("num_images", "1"))

        extra_params = {}
        if background != "否":
            extra_params["background"] = "transparent"
            extra_params["output_format"] = "png"
        if quality:
            quality = self.QUALITY_VALUE_MAP.get(quality, quality)
            extra_params["quality"] = quality

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
                    **extra_params,
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

        size_note = ""
        if aspect_ratio:
            size_note = f" | aspectRatio: {aspect_ratio}"
            if self.HAS_IMAGE_SIZE:
                size_note += f" | imageSize: {image_size}"
        failed_count = max(0, num_images - len(pil_images))
        fail_note = f" | 失败: {failed_count} 张" if failed_count > 0 else ""
        status = f"GPT Image | 模型: {model}{size_note} | 参考图片: {len(image_data_urls)} 张 | 成功生成: {len(pil_images)} 张{fail_note}"

        return {
            "ui": {"string": [status]},
            "result": (pil_to_tensor(pil_images), status),
        }


class GrsaiGPTImageFlareSunburst_Node(GrsaiGPTImage_Node):
    """GPT Image 2.5 flare / sunburst 节点"""

    MODELS = ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"]
    DEFAULT_MODEL = "gpt-image-2.5-flare"
    HAS_ASPECT_RATIO = False
    HAS_IMAGE_SIZE = True
    VIP = True
    BACKGROUND_OPTIONS = ["否", "是"]
    QUALITY_OPTIONS = [
        "auto",
        "low",
        "medium",
        "high",
        "xhigh（sunburst可用）",
        "max（sunburst可用）",
    ]
    QUALITY_VALUE_MAP = {
        "xhigh（sunburst可用）": "xhigh",
        "max（sunburst可用）": "max",
    }


class GrsaiGPTImageVIP_Node(GrsaiGPTImage_Node):
    """GPT Image 2 VIP 节点"""

    MODELS = ["gpt-image-2-vip"]
    DEFAULT_MODEL = "gpt-image-2-vip"
    HAS_ASPECT_RATIO = True
    HAS_IMAGE_SIZE = True
    VIP = True
    ASPECT_RATIO_OPTIONS = GPT_IMAGE_VIP_ASPECT_RATIO_OPTIONS
    BACKGROUND_OPTIONS = None


NODE_CLASS_MAPPINGS = {
    "Grsai_GPTImage": GrsaiGPTImage_Node,
    "Grsai_GPTImageFlareSunburst": GrsaiGPTImageFlareSunburst_Node,
    "Grsai_GPTImageVIP": GrsaiGPTImageVIP_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Grsai_GPTImage": "🎨 GrsAI GPT Image 2/2.5",
    "Grsai_GPTImageFlareSunburst": "🎨 GrsAI GPT Image 2.5 flare/sunburst",
    "Grsai_GPTImageVIP": "🎨 GrsAI GPT Image 2 VIP",
}
