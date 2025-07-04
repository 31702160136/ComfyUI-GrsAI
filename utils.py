"""
工具函数模块
提供图像处理、URL解析、数据转换等通用功能
"""

import io
import requests
import numpy as np
from PIL import Image
from typing import Optional, Union, List, Tuple
import torch
import re
from urllib.parse import urlparse


def download_image(url: str, timeout: int = 30) -> Optional[Image.Image]:
    """
    从URL下载图像

    Args:
        url: 图像URL
        timeout: 超时时间（秒）

    Returns:
        PIL.Image对象，如果下载失败返回None
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        print(f"图像下载失败，错误: {str(e)}")
        return None


def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """将torch张量（B, H, W, C）转换为PIL图像列表，支持RGBA透明通道"""
    if not isinstance(tensor, torch.Tensor):
        return []

    images = []
    for i in range(tensor.shape[0]):
        # [H, W, C]
        img_tensor = tensor[i]

        # 确保值在[0, 1]范围内
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # 转换为numpy数组并缩放到[0, 255]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)

        # 根据通道数创建相应格式的PIL图像
        if img_tensor.shape[2] == 4:  # RGBA
            images.append(Image.fromarray(img_np, "RGBA"))
        else:  # RGB
            images.append(Image.fromarray(img_np, "RGB"))

    return images


def handle_transparent_background(
    image: Image.Image, background_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    处理透明背景的图像

    Args:
        image: PIL图像对象
        background_color: 背景颜色RGB元组，默认为黑色(0, 0, 0)

    Returns:
        处理后的RGB图像
    """
    if image.mode == "RGBA":
        # 创建指定颜色的背景
        background = Image.new("RGB", image.size, background_color)
        # 使用alpha通道进行合成
        image = Image.alpha_composite(background.convert("RGBA"), image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    return image


def pil_to_tensor(
    pil_images: Union[Image.Image, List[Image.Image]],
    background_color: Union[Tuple[int, int, int], bool, None] = None,
    preserve_transparency: Optional[bool] = None,
) -> torch.Tensor:
    """
    将单个PIL图像或PIL图像列表转换为ComfyUI图像张量

    Args:
        pil_images: PIL图像或PIL图像列表
        background_color: 向后兼容参数 - 透明背景替换颜色，如果为tuple则不保留透明度
        preserve_transparency: 是否保留透明度信息，默认为True（除非指定了background_color）

    Returns:
        ComfyUI张量格式的图像
    """
    # 向后兼容性处理
    if background_color is not None and isinstance(background_color, tuple):
        # 旧API调用：指定了背景颜色，不保留透明度
        preserve_transparency = False
        bg_color = background_color
    else:
        # 新API调用：默认保留透明度
        if preserve_transparency is None:
            preserve_transparency = True
        bg_color = (0, 0, 0)  # 默认黑色背景
    if not isinstance(pil_images, list):
        pil_images = [pil_images]

    tensors = []
    for pil_image in pil_images:
        # 如果保留透明度且图像有alpha通道，则保持RGBA格式
        if preserve_transparency and pil_image.mode == "RGBA":
            # 保持RGBA格式
            processed_image = pil_image
        elif preserve_transparency and pil_image.mode in ("LA", "P"):
            # 将其他带透明度的格式转换为RGBA
            processed_image = pil_image.convert("RGBA")
        else:
            # 对于其他情况，转换为RGB（保持原有行为）
            if pil_image.mode == "RGBA":
                processed_image = handle_transparent_background(pil_image, bg_color)
            elif pil_image.mode != "RGB":
                processed_image = pil_image.convert("RGB")
            else:
                processed_image = pil_image

        img_array = np.array(processed_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array)[None,]
        tensors.append(tensor)

    if not tensors:
        # 如果列表为空，返回一个空的占位符张量
        channels = (
            4
            if (pil_images and pil_images[0].mode == "RGBA" and preserve_transparency)
            else 3
        )
        return torch.empty((0, 1, 1, channels), dtype=torch.float32)

    return torch.cat(tensors, dim=0)


def tensor_to_base64(tensor: torch.Tensor, image_format: str = "png") -> str:
    """
    将ComfyUI图像张量转换为Base64编码的字符串

    Args:
        tensor: ComfyUI图像张量
        image_format: 图像格式 ('png' or 'jpeg')

    Returns:
        str: Base64编码的字符串
    """
    import base64

    pil_images = tensor_to_pil(tensor)
    if not pil_images:
        raise ValueError("无法从张量转换为PIL图像")

    # 使用第一个图像进行Base64编码
    pil_image = pil_images[0]
    buffered = io.BytesIO()

    # 根据指定的格式保存
    if image_format.lower() == "jpeg":
        pil_image.save(buffered, format="JPEG")
    else:
        pil_image.save(buffered, format="PNG")

    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_string


def validate_aspect_ratio(aspect_ratio: str) -> bool:
    """
    验证宽高比格式是否正确

    Args:
        aspect_ratio: 宽高比字符串，如 "16:9"

    Returns:
        bool: 格式是否正确
    """
    pattern = r"^\d+:\d+$"
    return bool(re.match(pattern, aspect_ratio))


def calculate_dimensions(aspect_ratio: str, base_size: int = 1024) -> Tuple[int, int]:
    """
    根据宽高比计算图像尺寸

    Args:
        aspect_ratio: 宽高比字符串，如 "16:9"
        base_size: 基础尺寸

    Returns:
        Tuple[int, int]: (宽度, 高度)
    """
    try:
        width_ratio, height_ratio = map(int, aspect_ratio.split(":"))

        # 计算实际尺寸，保持总像素数接近base_size^2
        total_ratio = width_ratio * height_ratio
        scale = (base_size * base_size / total_ratio) ** 0.5

        width = int(width_ratio * scale)
        height = int(height_ratio * scale)

        # 确保尺寸是8的倍数（AI模型通常需要）
        width = (width // 8) * 8
        height = (height // 8) * 8

        return width, height
    except:
        return base_size, base_size


def format_error_message(error: Exception, context: str = "") -> str:
    """
    格式化错误消息，提供用户友好的错误信息

    Args:
        error: 异常对象
        context: 错误上下文

    Returns:
        str: 格式化的错误消息
    """
    error_type = type(error).__name__
    error_msg = str(error)

    if context:
        return f"[{context}] {error_type}: {error_msg}"
    else:
        return f"{error_type}: {error_msg}"


def safe_filename(filename: str) -> str:
    """
    生成安全的文件名，移除特殊字符

    Args:
        filename: 原始文件名

    Returns:
        str: 安全的文件名
    """
    # 移除或替换不安全的字符
    safe_chars = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # 限制长度
    if len(safe_chars) > 100:
        safe_chars = safe_chars[:100]

    return safe_chars


def bytes_to_mb(bytes_size: int) -> float:
    """
    将字节转换为MB

    Args:
        bytes_size: 字节大小

    Returns:
        float: MB大小
    """
    return bytes_size / (1024 * 1024)
