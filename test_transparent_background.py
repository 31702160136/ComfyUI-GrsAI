#!/usr/bin/env python3
"""
测试透明背景图片处理的脚本
"""

import os
import numpy as np
from PIL import Image
from utils import pil_to_tensor, handle_transparent_background, tensor_to_pil


def create_test_transparent_image():
    """创建一个测试用的透明背景图片"""
    # 创建一个100x100的RGBA图像
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))  # 完全透明

    # 在中心画一个红色的圆
    import PIL.ImageDraw as ImageDraw

    draw = ImageDraw.Draw(img)
    draw.ellipse([25, 25, 75, 75], fill=(255, 0, 0, 255))  # 红色圆形

    return img


def test_transparent_handling():
    """测试透明背景处理"""
    print("=== 透明背景处理测试 ===")

    # 创建测试图片
    test_img = create_test_transparent_image()
    print(f"原始图片模式: {test_img.mode}")

    # 测试不同背景颜色的处理
    background_colors = [
        (0, 0, 0),  # 黑色
        (255, 255, 255),  # 白色
        (128, 128, 128),  # 灰色
    ]

    for i, bg_color in enumerate(background_colors):
        print(f"\n测试背景颜色: {bg_color}")

        # 使用新的处理函数
        processed_img = handle_transparent_background(test_img, bg_color)
        print(f"处理后图片模式: {processed_img.mode}")

        # 转换为tensor再转回PIL来验证
        tensor = pil_to_tensor(test_img, bg_color)
        print(f"张量形状: {tensor.shape}")

        result_imgs = tensor_to_pil(tensor)
        print(f"转换回PIL图片数量: {len(result_imgs)}")

        # 保存测试结果
        output_path = (
            f"test_transparent_bg_{bg_color[0]}_{bg_color[1]}_{bg_color[2]}.png"
        )
        processed_img.save(output_path)
        print(f"保存测试图片: {output_path}")


def test_old_vs_new_behavior():
    """对比旧版本与新版本的行为差异"""
    print("\n=== 对比旧版本与新版本行为 ===")

    test_img = create_test_transparent_image()

    # 模拟旧版本行为（直接convert RGB）
    old_behavior = test_img.convert("RGB")
    old_behavior.save("test_old_behavior.png")
    print("旧版本行为 - 直接convert('RGB'): 保存为 test_old_behavior.png")

    # 新版本行为（黑色背景）
    new_behavior_black = handle_transparent_background(test_img, (0, 0, 0))
    new_behavior_black.save("test_new_behavior_black.png")
    print("新版本行为 - 黑色背景: 保存为 test_new_behavior_black.png")

    # 新版本行为（白色背景）
    new_behavior_white = handle_transparent_background(test_img, (255, 255, 255))
    new_behavior_white.save("test_new_behavior_white.png")
    print("新版本行为 - 白色背景: 保存为 test_new_behavior_white.png")


if __name__ == "__main__":
    test_transparent_handling()
    test_old_vs_new_behavior()

    print("\n=== 测试完成 ===")
    print("请检查生成的测试图片，对比透明背景处理效果。")
    print("新版本应该不会出现泛光问题。")
