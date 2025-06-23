#!/usr/bin/env python3
"""
快速测试 generate_single_image 函数
简化版本，专注于核心功能测试
"""

import os
import sys
import time
import random

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def quick_test():
    """快速测试函数"""
    try:
        from api_client import GrsaiAPI, GrsaiAPIError
        from config import default_config
        from utils import format_error_message
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

    # 检查API密钥
    api_key = default_config.get_api_key()
    if not api_key:
        print("❌ 未找到API密钥，请先配置 GRSAI_API_KEY 环境变量")
        print(default_config.api_key_error_message)
        return False

    print("🚀 快速测试 generate_single_image 函数")
    print("=" * 50)

    # 定义内部函数（从 nodes.py 提取）
    def generate_single_image(current_seed):
        try:
            api_client = FluxKontextAPI(api_key=api_key)
            api_params = {
                "prompt": "A beautiful landscape with mountains and a lake",
                "model": "flux-kontext-max",
                "seed": current_seed,
                "aspect_ratio": "1:1",
            }
            pil_image, url = api_client.generate_image(**api_params)
            return pil_image, url
        except Exception as e:
            return e

    # 执行测试
    test_seed = random.randint(1, 2147483647)
    print(f"🎲 使用随机种子: {test_seed}")
    print("⏳ 正在生成图像...")

    start_time = time.time()
    result = generate_single_image(test_seed)
    duration = time.time() - start_time

    if isinstance(result, Exception):
        print(f"❌ 测试失败: {format_error_message(result)}")
        print(f"⏱️  耗时: {duration:.2f}秒")
        return False
    else:
        pil_image, url = result
        print(f"✅ 测试成功!")
        print(f"🖼️  图像尺寸: {pil_image.size}")
        print(f"🔗 图像URL: {url}")
        print(f"⏱️  耗时: {duration:.2f}秒")

        # 保存测试图像
        save_path = f"quick_test_{test_seed}.png"
        pil_image.save(save_path)
        print(f"💾 图像已保存: {save_path}")
        return True


if __name__ == "__main__":
    success = quick_test()
    print(f"\n🏁 测试结果: {'通过' if success else '失败'}")
