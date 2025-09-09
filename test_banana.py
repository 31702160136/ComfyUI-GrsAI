#!/usr/bin/env python3
"""
Nano Banana API 简单测试
 - 测试默认模型: nano-banana
 - 测试加速模型: nano-banana-fast

运行: python test_banana.py
"""

import os
import sys
import time
from typing import Tuple

# 保证可从当前目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from api_client import GrsaiAPI, GrsaiAPIError
    from config import default_config
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)


# 统一测试输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_case(model: str, prompt: str) -> bool:
    api_key = default_config.get_api_key()
    if not api_key:
        print(default_config.api_key_error_message)
        return False

    print(f"\n=== 测试模型: {model} ===")
    try:
        client = GrsaiAPI(api_key=api_key)
        start = time.time()
        pil_images, image_urls, errors = client.banana_generate_image(
            prompt=prompt, model=model, urls=[]
        )
        duration = time.time() - start

        if errors:
            print(f"❌ API 返回错误: {errors}")
            print(f"⏱️ 耗时: {duration:.2f}s")
            return False

        if not pil_images:
            print("❌ 未返回任何图像")
            return False

        print(f"✅ 成功生成 {len(pil_images)} 张图像 | ⏱️ {duration:.2f}s")

        # 保存图像
        for idx, img in enumerate(pil_images, start=1):
            save_path = os.path.join(OUTPUT_DIR, f"banana_{model}_{idx}.png")
            try:
                img.save(save_path)
                print(f"💾 已保存: {save_path}")
            except Exception as e:
                print(f"⚠️ 保存图像失败: {e}")
        return True

    except GrsaiAPIError as e:
        print(f"❌ API 错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知异常: {e}")
        return False


def main() -> int:
    print("🚀 开始测试 Nano Banana API")
    prompt = (
        "Create a high-quality studio shot of a ripe banana on a matte"
        " surface, soft shadows, natural lighting."
    )

    cases = [
        # ("nano-banana", prompt),
        ("nano-banana-fast", prompt),
    ]

    passed = 0
    for model, p in cases:
        if run_case(model, p):
            passed += 1

    total = len(cases)
    print("\n=== 测试总结 ===")
    print(f"通过: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

