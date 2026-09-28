"""
工具函数模块
提供图像处理、URL解析、数据转换等通用功能
"""

import io
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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


def _load_preview_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """加载跨平台、尽可能支持中文的预览字体。"""
    font_names = (
        [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
        if bold
        else [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
    )
    fallback_names = (
        ["Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["Arial.ttf", "DejaVuSans.ttf"]
    )
    for font_name in [*font_names, *fallback_names]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except (OSError, ValueError):
            continue

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _wrap_preview_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> List[str]:
    """按实际像素宽度换行，长错误文本会截断并显示省略号。"""
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        return ["未提供具体错误信息"]

    lines: List[str] = []
    current = ""
    truncated = False
    for char in normalized:
        candidate = current + char
        box = draw.textbbox((0, 0), candidate, font=font)
        if current and box[2] - box[0] > max_width:
            lines.append(current.rstrip())
            current = char.lstrip()
            if len(lines) == max_lines:
                truncated = True
                break
        else:
            current = candidate

    if len(lines) < max_lines and current:
        lines.append(current.rstrip())

    if truncated and lines:
        ellipsis = "…"
        last_line = lines[-1]
        while last_line:
            candidate = last_line.rstrip() + ellipsis
            box = draw.textbbox((0, 0), candidate, font=font)
            if box[2] - box[0] <= max_width:
                lines[-1] = candidate
                break
            last_line = last_line[:-1]
        if not last_line:
            lines[-1] = ellipsis

    return lines


def create_generation_failed_image(
    width: int = 1024,
    height: int = 1024,
    error_message: str = "",
) -> Image.Image:
    """创建包含实际失败原因、适合在 ComfyUI 中预览的提示图。"""
    width = max(320, int(width))
    height = max(320, int(height))
    image = Image.new("RGB", (width, height), (35, 24, 24))
    draw = ImageDraw.Draw(image)
    margin = max(24, min(width, height) // 12)
    line_width = max(8, min(width, height) // 80)
    red = (220, 68, 68)

    draw.rounded_rectangle(
        (margin, margin, width - margin, height - margin),
        radius=max(16, margin // 2),
        outline=red,
        width=line_width,
    )

    center_x = width // 2
    cross_center_y = margin + max(72, min(width, height) // 7)
    cross_size = max(38, min(width, height) // 13)
    draw.line(
        (
            center_x - cross_size,
            cross_center_y - cross_size,
            center_x + cross_size,
            cross_center_y + cross_size,
        ),
        fill=red,
        width=line_width * 2,
    )
    draw.line(
        (
            center_x + cross_size,
            cross_center_y - cross_size,
            center_x - cross_size,
            cross_center_y + cross_size,
        ),
        fill=red,
        width=line_width * 2,
    )

    scale = min(width, height) / 1024
    title_font = _load_preview_font(max(32, round(54 * scale)), bold=True)
    label_font = _load_preview_font(max(26, round(36 * scale)), bold=True)
    reason_font = _load_preview_font(max(24, round(32 * scale)))

    title = "生成失败 / GENERATION FAILED"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    if title_box[2] - title_box[0] > width - 2 * margin:
        title = "GENERATION FAILED"
        title_box = draw.textbbox((0, 0), title, font=title_font)
    if title_box[2] - title_box[0] > width - 2 * margin:
        title = "FAILED"
        title_box = draw.textbbox((0, 0), title, font=title_font)
    title_y = cross_center_y + cross_size + max(28, round(34 * scale))
    draw.text(
        ((width - (title_box[2] - title_box[0])) // 2, title_y),
        title,
        fill=(255, 225, 225),
        font=title_font,
    )

    content_left = margin + max(24, round(28 * scale))
    content_width = width - 2 * content_left
    label_y = title_y + (title_box[3] - title_box[1]) + max(34, round(48 * scale))
    draw.text(
        (content_left, label_y),
        "失败原因：",
        fill=(245, 170, 170),
        font=label_font,
    )

    label_box = draw.textbbox((0, 0), "失败原因：", font=label_font)
    reason_y = label_y + (label_box[3] - label_box[1]) + max(16, round(20 * scale))
    line_spacing = max(10, round(12 * scale))
    line_box = draw.textbbox((0, 0), "Ag中文", font=reason_font)
    line_height = max(1, line_box[3] - line_box[1]) + line_spacing
    available_height = max(line_height, height - margin - reason_y)
    max_lines = max(1, available_height // line_height)
    reason_lines = _wrap_preview_text(
        draw,
        error_message or "图像生成失败，服务未提供具体错误信息",
        reason_font,
        content_width,
        max_lines,
    )
    draw.multiline_text(
        (content_left, reason_y),
        "\n".join(reason_lines),
        fill=(245, 225, 225),
        font=reason_font,
        spacing=line_spacing,
    )
    return image


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

    processed_images: List[Image.Image] = []
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

        processed_images.append(processed_image)

    if not processed_images:
        # 如果列表为空，返回一个空的占位符张量
        channels = (
            4
            if (pil_images and pil_images[0].mode == "RGBA" and preserve_transparency)
            else 3
        )
        return torch.empty((0, 1, 1, channels), dtype=torch.float32)

    # 批次内若存在多种尺寸/通道，统一对齐到最大尺寸 + 最大通道数
    # 通过“居中填充”而非缩放，避免内容失真；
    # RGBA 用透明填充，RGB 用 bg_color 填充。
    max_w = max(img.width for img in processed_images)
    max_h = max(img.height for img in processed_images)
    has_rgba = any(img.mode == "RGBA" for img in processed_images)
    target_mode = "RGBA" if has_rgba else "RGB"
    if target_mode == "RGBA":
        fill_color: Tuple[int, ...] = (0, 0, 0, 0)
    else:
        fill_color = bg_color

    aligned_images: List[Image.Image] = []
    for img in processed_images:
        if img.mode != target_mode:
            img = img.convert(target_mode)
        if img.size == (max_w, max_h):
            aligned_images.append(img)
            continue
        canvas = Image.new(target_mode, (max_w, max_h), fill_color)
        offset = ((max_w - img.width) // 2, (max_h - img.height) // 2)
        if target_mode == "RGBA":
            canvas.paste(img, offset, img)
        else:
            canvas.paste(img, offset)
        aligned_images.append(canvas)

    tensors = []
    for img in aligned_images:
        img_array = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array)[None,]
        tensors.append(tensor)

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
