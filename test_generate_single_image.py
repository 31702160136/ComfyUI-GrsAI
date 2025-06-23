#!/usr/bin/env python3
"""
测试 generate_single_image 函数的本地测试用例
使用方法: python test_generate_single_image.py
"""

import os
import sys
import time
import random
import tempfile
from pathlib import Path
from typing import Optional, Any

# 添加当前目录到Python路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from api_client import GrsaiAPI, GrsaiAPIError
    from config import default_config
    from utils import download_image, pil_to_tensor, format_error_message, tensor_to_pil
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有依赖的模块都在当前目录中")
    sys.exit(1)


def test_generate_single_image_function():
    """
    测试 generate_single_image 函数（从 nodes.py 中提取）
    """

    def generate_single_image(
        grsai_api_key: str, final_prompt: str, current_seed: int, model: str, **kwargs
    ):
        """
        从 nodes.py 中提取的 generate_single_image 函数
        """
        try:
            api_client = FluxKontextAPI(api_key=grsai_api_key)
            api_params = {
                "prompt": final_prompt,
                "model": model,
                "seed": current_seed,
            }
            api_params.update(kwargs)
            pil_image, url = api_client.generate_image(**api_params)
            return pil_image, url
        except Exception as e:
            return e

    # 测试参数配置
    test_cases = [
        {
            "name": "基础文本生图测试",
            "prompt": "A beautiful sunset over the mountains",
            "model": "flux-kontext-max",
            "seed": 12345,
            "aspect_ratio": "1:1",
        },
        {
            "name": "随机种子测试",
            "prompt": "A cute cat sitting on a chair",
            "model": "flux-kontext-pro",
            "seed": random.randint(1, 2147483647),
            "aspect_ratio": "16:9",
        },
        {
            "name": "复杂提示词测试",
            "prompt": 'Extreme close-up of a single tiger eye, direct frontal view. Detailed iris and pupil. Sharp focus on eye texture and color. Natural lighting to capture authentic eye shine and depth. The word "FLUX" is painted over it in big, white brush strokes with visible texture.',
            "model": "flux-kontext-max",
            "seed": 0,  # 将自动生成随机种子
            "aspect_ratio": "4:3",
        },
    ]

    print("=" * 60)
    print("开始测试 generate_single_image 函数")
    print("=" * 60)

    # 检查API密钥
    grsai_api_key = default_config.get_api_key()
    if not grsai_api_key:
        print("❌ 错误: 未找到API密钥")
        print(default_config.api_key_error_message)
        return False

    print(f"✅ API密钥已加载: {grsai_api_key[:10]}...")

    success_count = 0
    total_tests = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试 {i}/{total_tests}: {test_case['name']}")
        print("-" * 40)

        # 处理种子值
        current_seed = test_case["seed"]
        if current_seed == 0:
            current_seed = random.randint(1, 2147483647)
            print(f"🎲 生成随机种子: {current_seed}")
        else:
            print(f"🔢 使用固定种子: {current_seed}")

        print(f"📝 提示词: {test_case['prompt'][:50]}...")
        print(f"🔧 模型: {test_case['model']}")
        print(f"📐 宽高比: {test_case['aspect_ratio']}")

        # 提取测试参数
        prompt = test_case["prompt"]
        model = test_case["model"]
        aspect_ratio = test_case["aspect_ratio"]

        print("⏳ 开始生成图像...")
        start_time = time.time()

        try:
            # 调用 generate_single_image 函数
            result = generate_single_image(
                grsai_api_key=grsai_api_key,
                final_prompt=prompt,
                current_seed=current_seed,
                model=model,
                aspect_ratio=aspect_ratio,
            )

            end_time = time.time()
            duration = end_time - start_time

            if isinstance(result, Exception):
                print(f"❌ 生成失败: {format_error_message(result)}")
                print(f"⏱️  耗时: {duration:.2f}秒")
            else:
                pil_image, url = result
                print(f"✅ 生成成功!")
                print(f"🖼️  图像尺寸: {pil_image.size}")
                print(f"🔗 图像URL: {url}")
                print(f"⏱️  耗时: {duration:.2f}秒")

                # 可选：保存图像到本地
                if pil_image:
                    save_path = f"test_output_{i}_{current_seed}.png"
                    pil_image.save(save_path)
                    print(f"💾 图像已保存到: {save_path}")

                success_count += 1

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"❌ 测试异常: {format_error_message(e)}")
            print(f"⏱️  耗时: {duration:.2f}秒")

    print("\n" + "=" * 60)
    print(f"测试完成! 成功: {success_count}/{total_tests}")
    print("=" * 60)

    return success_count == total_tests


def test_error_scenarios():
    """
    测试错误场景
    """

    def generate_single_image(
        grsai_api_key: str, final_prompt: str, current_seed: int, model: str, **kwargs
    ):
        try:
            api_client = FluxKontextAPI(api_key=grsai_api_key)
            api_params = {
                "prompt": final_prompt,
                "model": model,
                "seed": current_seed,
            }
            api_params.update(kwargs)
            pil_image, url = api_client.generate_image(**api_params)
            return pil_image, url
        except Exception as e:
            return e

    print("\n" + "=" * 60)
    print("开始测试错误场景")
    print("=" * 60)

    error_test_cases = [
        {
            "name": "无效API密钥测试",
            "api_key": "invalid_api_key_123",
            "prompt": "Test prompt",
            "model": "flux-kontext-max",
            "seed": 12345,
            "expected_error": "API密钥无效",
        },
        {
            "name": "空提示词测试",
            "api_key": default_config.get_api_key(),
            "prompt": "",
            "model": "flux-kontext-max",
            "seed": 12345,
            "expected_error": "提示词不能为空",
        },
        {
            "name": "无效模型测试",
            "api_key": default_config.get_api_key(),
            "prompt": "Test prompt",
            "model": "invalid_model",
            "seed": 12345,
            "expected_error": "无效的模型",
        },
    ]

    for i, test_case in enumerate(error_test_cases, 1):
        print(f"\n🧪 错误测试 {i}: {test_case['name']}")
        print("-" * 40)

        try:
            result = generate_single_image(
                grsai_api_key=test_case["api_key"],
                final_prompt=test_case["prompt"],
                current_seed=test_case["seed"],
                model=test_case["model"],
            )

            if isinstance(result, Exception):
                print(f"✅ 预期错误: {format_error_message(result)}")
            else:
                print(f"⚠️  意外成功: 预期应该失败")

        except Exception as e:
            print(f"✅ 捕获到异常: {format_error_message(e)}")


def main():
    """
    主测试函数
    """
    print("🚀 Flux-Kontext generate_single_image 函数测试")
    print(f"📁 当前工作目录: {os.getcwd()}")
    print(f"🐍 Python版本: {sys.version}")

    # 检查必要的环境
    try:
        import requests

        print("✅ requests 模块已安装")
    except ImportError:
        print("❌ requests 模块未安装，请运行: pip install requests")
        return

    try:
        from PIL import Image

        print("✅ PIL 模块已安装")
    except ImportError:
        print("❌ PIL 模块未安装，请运行: pip install Pillow")
        return

    # 运行功能测试
    success = test_generate_single_image_function()

    # 运行错误场景测试（仅在有API密钥时）
    if default_config.get_api_key():
        test_error_scenarios()
    else:
        print("\n⚠️  跳过错误场景测试（需要API密钥）")

    print(f"\n🏁 测试结束，结果: {'通过' if success else '失败'}")


if __name__ == "__main__":
    main()
