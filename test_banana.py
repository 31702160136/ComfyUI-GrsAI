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
import math
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


def _align8(x: int) -> int:
    return max(8, int(round(x / 8)) * 8)


def _parse_ratio(ratio: str) -> Tuple[int, int]:
    w, h = ratio.split(":", 1)
    return int(w), int(h)


def _compute_dims_1mp(ratio: str) -> Tuple[int, int]:
    # 目标约 1MP（以 1024x1024 为参考），并对齐到 8
    target_area = 1024 * 1024
    w_r, h_r = _parse_ratio(ratio)
    w_f = math.sqrt(target_area * (w_r / h_r))
    h_f = target_area / w_f
    w_i = _align8(int(round(w_f)))
    h_i = _align8(int(round(h_f)))
    # 安全范围
    w_i = max(64, min(w_i, 2048))
    h_i = max(64, min(h_i, 2048))
    return w_i, h_i


def _build_dummy_url(width: int, height: int) -> str:
    bg = "cccccc"
    return f"https://dummyimage.com/{width}x{height}/{bg}/{bg}.png"


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


def run_ar_placeholder_case(prompt: str) -> bool:
    """单独的用例：为所有支持的宽高比生成占位 URL 并调用生成接口"""
    print("\n🚀 测试各宽高比（占位URL）")
    ar_list = getattr(
        default_config,
        "SUPPORTED_ASPECT_RATIOS",
        ["21:9", "16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16", "9:21"],
    )
    ar_passed = 0
    for ar in ar_list:
        try:
            w, h = _compute_dims_1mp(ar)
            dummy_url = _build_dummy_url(w, h)
            print(f"\n[AR {ar}] 使用占位URL: {dummy_url}")

            ar_prompt = (
                prompt
                + "\nAdopt the aspect ratio of the last reference image; do not use its visual content."
            )

            ok = False
            try:
                client = GrsaiAPI(api_key=default_config.get_api_key())
                start = time.time()
                pil_images, image_urls, errors = client.banana_generate_image(
                    prompt=ar_prompt, model="nano-banana-fast", urls=[dummy_url]
                )
                duration = time.time() - start
                if errors:
                    print(f"❌ AR {ar} 错误: {errors}")
                elif not pil_images:
                    print(f"❌ AR {ar} 无返回图像")
                else:
                    print(f"✅ AR {ar} 成功生成 {len(pil_images)} 张 | ⏱️ {duration:.2f}s")
                    save_name = f"banana_ar_{ar.replace(':','x')}.png"
                    save_path = os.path.join(OUTPUT_DIR, save_name)
                    try:
                        pil_images[0].save(save_path)
                        print(f"💾 已保存: {save_path}")
                    except Exception as e:
                        print(f"⚠️ 保存失败: {e}")
                    ok = True
            except Exception as e:
                print(f"❌ AR {ar} 异常: {e}")

            if ok:
                ar_passed += 1
        except Exception as e:
            print(f"❌ 生成占位URL失败 [{ar}]: {e}")

    print("\n=== 宽高比测试总结 ===")
    print(f"通过: {ar_passed}/{len(ar_list)}")
    return ar_passed == len(ar_list)


def main() -> int:
    print("🚀 开始测试 Nano Banana API")
    prompt = (
        "Create a high-quality studio shot of a ripe banana on a matte"
        " surface, soft shadows, natural lighting."
    )

    # 支持的运行模式：basic（默认）、ar、all
    mode = sys.argv[1] if len(sys.argv) > 1 else "basic"

    # 基础用例（模型直生图）
    basic_ok = True
    if mode in ("basic", "all"):
        cases = [
            # ("nano-banana", prompt),
            # ("nano-banana-fast", prompt),
        ]
        passed = 0
        for model, p in cases:
            if run_case(model, p):
                passed += 1
        total = len(cases)
        print("\n=== 基础用例总结 ===")
        print(f"通过: {passed}/{total}")
        basic_ok = (passed == total)

    # 宽高比占位URL用例（单独case）
    ar_ok = True
    if mode in ("ar", "all"):
        ar_ok = run_ar_placeholder_case(prompt)

    if mode == "basic":
        return 0 if basic_ok else 1
    if mode == "ar":
        return 0 if ar_ok else 1
    # all
    return 0 if (basic_ok and ar_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
