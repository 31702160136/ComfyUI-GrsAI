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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        print(f"图像下载失败，错误: {str(e)}")
        return None

def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """将torch张量（B, H, W, C）转换为PIL图像列表，使其更健壮"""
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
        
        # 创建PIL图像
        images.append(Image.fromarray(img_np, 'RGB'))
        
    return images

def pil_to_tensor(pil_images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """
    将单个PIL图像或PIL图像列表转换为ComfyUI图像张量
    """
    if not isinstance(pil_images, list):
        pil_images = [pil_images]

    tensors = []
    for pil_image in pil_images:
        # 确保图像是RGB格式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array)[None,]
        tensors.append(tensor)
    
    if not tensors:
        # 如果列表为空，返回一个空的占位符张量
        return torch.empty((0, 1, 1, 3), dtype=torch.float32)
        
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
    if image_format.lower() == 'jpeg':
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
    pattern = r'^\d+:\d+$'
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
        width_ratio, height_ratio = map(int, aspect_ratio.split(':'))
        
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
    safe_chars = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
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