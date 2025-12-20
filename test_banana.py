#!/usr/bin/env python3
"""
Nano Banana API 简单测试
- 测试默认模型: nano-banana
- 测试加速模型: nano-banana-fast
- 测试专业模型: nano-banana-pro（支持 imageSize）
- 测试专业模型: nano-banana-pro-vt（支持 imageSize）

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


def _align32_floor(x: int) -> int:
    """向下取整到最接近的32倍，并保证至少为32"""
    if x <= 0:
        return 32
    aligned = (x // 32) * 32
    return aligned if aligned >= 32 else 32


def _parse_ratio(ratio: str) -> Tuple[int, int]:
    w, h = ratio.split(":", 1)
    return int(w), int(h)


def _compute_dims_1mp(ratio: str) -> Tuple[int, int]:
    # 目标约 1MP（以 1024x1024 为参考），再向下对齐到 32 像素网格
    target_area = 1024 * 1024
    w_r, h_r = _parse_ratio(ratio)
    w_f = math.sqrt(target_area * (w_r / h_r))
    h_f = target_area / w_f
    w_i = _align32_floor(int(round(w_f)))
    h_i = _align32_floor(int(round(h_f)))

    # API 返回的最大边约 1536，且最小不低于 64
    w_i = max(64, min(w_i, 1536))
    h_i = max(64, min(h_i, 1536))
    return w_i, h_i


def _build_dummy_url(width: int, height: int) -> str:
    bg = "cccccc"
    return f"https://dummyimage.com/{width}x{height}/{bg}/{bg}.png"


def run_case(model: str, prompt: str, aspect_ratio: str = "auto", image_size: str = None) -> bool:
    api_key = default_config.get_api_key()
    if not api_key:
        print(default_config.api_key_error_message)
        return False

    print(f"\n=== 测试模型: {model} ===")
    try:
        client = GrsaiAPI(api_key=api_key)
        start = time.time()
        print(f"📐 宽高比: {aspect_ratio}")
        pil_images, image_urls, errors = client.banana_generate_image(
            prompt=prompt, model=model, urls=[], aspect_ratio=aspect_ratio, image_size=image_size
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
    """单独的用例：遍历全部支持的宽高比并调用生成接口"""
    print("\n🚀 测试各宽高比（API 参数）")
    ar_list = getattr(
        default_config,
        "SUPPORTED_NANO_BANANA_AR",
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
        ],
    )
    ar_passed = 0
    for ar in ar_list:
        try:
            if ":" in ar:
                w, h = _compute_dims_1mp(ar)
                print(f"\n[AR {ar}] 目标尺寸约: {w}x{h}")
            else:
                print(f"\n[AR {ar}] 使用自动宽高比")

            ok = False
            try:
                client = GrsaiAPI(api_key=default_config.get_api_key())
                start = time.time()
                pil_images, image_urls, errors = client.banana_generate_image(
                    prompt,
                    model="nano-banana-fast",
                    urls=[],
                    aspect_ratio=ar,
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


def run_invalid_image_size_case(prompt: str) -> bool:
    """验证 imageSize 行为（本地校验）：
    - PRO / PRO-VT：非法 imageSize 会在本地校验阶段拦截
    """
    api_key = default_config.get_api_key()
    if not api_key:
        print(default_config.api_key_error_message)
        return False

    print("\n🧪 校验 imageSize 约束")
    client = GrsaiAPI(api_key=api_key)

    # PRO 模型非法尺寸应抛错
    try:
        client.banana_generate_image(prompt=prompt, model="nano-banana-pro", image_size="8K")
        print("❌ 预期失败的调用却成功了（非法 imageSize 未被拦截）")
        return False
    except GrsaiAPIError as e:
        print(f"✅ 非法 imageSize 拒绝成功: {e}")

    # PRO-VT 模型非法尺寸应抛错
    try:
        client.banana_generate_image(prompt=prompt, model="nano-banana-pro-vt", image_size="8K")
        print("❌ 预期失败的调用却成功了（非法 imageSize 未被拦截）")
        return False
    except GrsaiAPIError as e:
        print(f"✅ 非法 imageSize 拒绝成功: {e}")

    return True


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
            ("nano-banana-fast", prompt),
            ("nano-banana-pro", prompt),
            ("nano-banana-pro-vt", prompt),
        ]
        passed = 0
        for model, p in cases:
            image_size = "1K" if model in ("nano-banana-pro", "nano-banana-pro-vt") else None
            if run_case(model, p, image_size=image_size):
                passed += 1
        total = len(cases)
        print("\n=== 基础用例总结 ===")
        print(f"通过: {passed}/{total}")
        basic_ok = (passed == total)

        # 本地校验 imageSize 合法性
        basic_ok = basic_ok and run_invalid_image_size_case(prompt)

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
