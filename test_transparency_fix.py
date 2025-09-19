#!/usr/bin/env python3
"""
测试透明背景修复的脚本
验证PNG图像的透明背景是否能正确保留
"""

import os
import numpy as np
from PIL import Image
from utils import pil_to_tensor, tensor_to_pil

# Ensure a dedicated directory for test outputs
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_transparent_test_image():
    """创建一个带透明背景的测试图像"""
    # 创建一个150x150的透明RGBA图像
    img = Image.new("RGBA", (150, 150), (0, 0, 0, 0))  # 完全透明背景

    # 画一个蓝色的圆
    import PIL.ImageDraw as ImageDraw

    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 110, 110], fill=(0, 100, 255, 255))  # 蓝色圆形

    # 画一些透明度不同的形状
    draw.rectangle([20, 20, 60, 60], fill=(255, 0, 0, 128))  # 半透明红色矩形
    draw.rectangle([90, 90, 130, 130], fill=(0, 255, 0, 200))  # 半透明绿色矩形

    return img


def test_transparency_preservation():
    """测试透明度保留功能"""
    print("=== 测试透明背景保留功能 ===")

    # 创建测试图像
    test_img = create_transparent_test_image()
    print(f"原始图像模式: {test_img.mode}")
    print(f"原始图像尺寸: {test_img.size}")

    # 保存原始测试图像
    test_img.save(os.path.join(OUTPUT_DIR, "original_transparent.png"))
    print("保存原始图像: original_transparent.png")

    # 测试1: 保留透明度（新功能）
    print("\n--- 测试1: 保留透明度 ---")
    tensor_rgba = pil_to_tensor(test_img, preserve_transparency=True)
    print(f"RGBA张量形状: {tensor_rgba.shape}")

    result_rgba = tensor_to_pil(tensor_rgba)
    if result_rgba:
        result_img = result_rgba[0]
        print(f"结果图像模式: {result_img.mode}")
        result_img.save(os.path.join(OUTPUT_DIR, "result_transparent_preserved.png"))
        print("保存结果: result_transparent_preserved.png")

    # 测试2: 使用旧API（向后兼容）
    print("\n--- 测试2: 旧API向后兼容 ---")
    tensor_rgb_black = pil_to_tensor(test_img, (0, 0, 0))  # 黑色背景
    print(f"RGB黑背景张量形状: {tensor_rgb_black.shape}")

    result_rgb_black = tensor_to_pil(tensor_rgb_black)
    if result_rgb_black:
        result_img_black = result_rgb_black[0]
        print(f"结果图像模式: {result_img_black.mode}")
        result_img_black.save(os.path.join(OUTPUT_DIR, "result_black_background.png"))
        print("保存结果: result_black_background.png")

    # 测试3: 白色背景
    print("\n--- 测试3: 白色背景 ---")
    tensor_rgb_white = pil_to_tensor(test_img, (255, 255, 255))  # 白色背景
    print(f"RGB白背景张量形状: {tensor_rgb_white.shape}")

    result_rgb_white = tensor_to_pil(tensor_rgb_white)
    if result_rgb_white:
        result_img_white = result_rgb_white[0]
        print(f"结果图像模式: {result_img_white.mode}")
        result_img_white.save(os.path.join(OUTPUT_DIR, "result_white_background.png"))
        print("保存结果: result_white_background.png")

    print("\n=== 测试完成 ===")
    print("请检查生成的图像文件:")
    print("- original_transparent.png: 原始透明图像")
    print("- result_transparent_preserved.png: 保留透明度的结果（应该保持透明背景）")
    print("- result_black_background.png: 黑色背景的结果")
    print("- result_white_background.png: 白色背景的结果")


if __name__ == "__main__":
    test_transparency_preservation()
