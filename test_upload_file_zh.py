#!/usr/bin/env python3
"""
测试 upload_file_zh 函数的测试用例
使用方法: python test_upload_file_zh.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path
import json

# 添加当前目录到Python路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尝试加载 .env 中的环境变量
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # 未安装 python-dotenv 或加载失败时忽略，按原逻辑读取环境变量
    pass

try:
    from upload import upload_file_zh, get_upload_token_zh
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保 upload.py 模块在当前目录中")
    sys.exit(1)


def create_test_image(file_path: str, size: tuple = (100, 100)):
    """
    创建一个测试用的图像文件

    Args:
        file_path (str): 文件路径
        size (tuple): 图像尺寸，默认 (100, 100)
    """
    try:
        from PIL import Image
        import io

        # 创建一个简单的彩色图像
        image = Image.new("RGB", size, color="red")

        # 在图像上添加一些图形
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image)
        draw.rectangle([10, 10, size[0] - 10, size[1] - 10], outline="blue", width=3)
        draw.text((20, 40), "TEST", fill="white")

        # 保存图像
        image.save(file_path, "PNG")
        return True

    except ImportError:
        # 如果PIL不可用，创建一个简单的文本文件作为替代
        with open(file_path, "w") as f:
            f.write("Test file content for upload")
        return False


def test_upload_file_zh_basic():
    """
    测试基本的文件上传功能
    """
    print("=" * 60)
    print("测试基本文件上传功能")
    print("=" * 60)

    # 检查API密钥
    api_key = ""
    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # 创建测试图像
        has_pil = create_test_image(temp_path)
        if has_pil:
            print(f"✅ 创建测试图像: {temp_path}")
        else:
            print(f"⚠️ 创建测试文件: {temp_path} (PIL不可用，使用文本文件)")

        print(f"📂 文件大小: {os.path.getsize(temp_path)} 字节")

        # 开始上传
        print("⏳ 开始上传文件...")
        start_time = time.time()

        result_url = upload_file_zh(temp_path)

        end_time = time.time()
        duration = end_time - start_time

        if result_url:
            print(f"✅ 上传成功!")
            print(f"🔗 文件URL: {result_url}")
            print(f"⏱️ 耗时: {duration:.2f}秒")
            return True
        else:
            print(f"❌ 上传失败: 返回空URL")
            return False

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ 上传异常: {e}")
        print(f"⏱️ 耗时: {duration:.2f}秒")
        return False

    finally:
        # 清理临时文件
        try:
            os.unlink(temp_path)
            print(f"🗑️ 已清理临时文件: {temp_path}")
        except:
            pass


def test_upload_file_zh_error_scenarios():
    """
    测试错误场景
    """
    print("\n" + "=" * 60)
    print("测试错误场景")
    print("=" * 60)

    success_count = 0
    total_tests = 0

    aa = {"prompt": "你会后悔会撒娇来上课垃圾费"}
    print(json.dumps(aa, indent=2))

    print(aa)

    # return
    # 测试1: 空文件路径
    print("\n📝 测试 1: 空文件路径")
    print("-" * 40)
    total_tests += 1

    try:
        result = upload_file_zh("")
        if result == "":
            print("✅ 空文件路径处理正确: 返回空字符串")
            success_count += 1
        else:
            print(f"❌ 空文件路径处理错误: 返回 {result}")
    except Exception as e:
        print(f"❌ 空文件路径测试异常: {e}")

    # 测试2: 文件不存在
    print("\n📝 测试 2: 文件不存在")
    print("-" * 40)
    total_tests += 1

    try:
        result = upload_file_zh("non_existent_file.png")
        print(f"❌ 应该抛出FileNotFoundError，但返回了: {result}")
    except FileNotFoundError as e:
        print(f"✅ 正确抛出FileNotFoundError: {e}")
        success_count += 1
    except Exception as e:
        print(f"❌ 抛出了错误的异常类型: {type(e).__name__}: {e}")

    # 测试3: 无效API密钥
    print("\n📝 测试 3: 无效API密钥")
    print("-" * 40)
    total_tests += 1

    # 备份原始API密钥
    original_api_key = os.getenv("GRSAI_API_KEY")

    try:
        # 设置无效的API密钥
        os.environ["GRSAI_API_KEY"] = "invalid_api_key_123"

        # 创建临时测试文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name

        create_test_image(temp_path)

        try:
            result = upload_file_zh(temp_path)
            print(f"❌ 应该抛出异常，但返回了: {result}")
        except Exception as e:
            print(f"✅ 正确抛出异常: {type(e).__name__}: {e}")
            success_count += 1
        finally:
            os.unlink(temp_path)

    finally:
        # 恢复原始API密钥
        if original_api_key:
            os.environ["GRSAI_API_KEY"] = original_api_key
        else:
            if "GRSAI_API_KEY" in os.environ:
                del os.environ["GRSAI_API_KEY"]

    print(f"\n错误场景测试完成: {success_count}/{total_tests} 通过")
    return success_count == total_tests


def test_different_file_types():
    """
    测试不同文件类型的上传
    """
    print("\n" + "=" * 60)
    print("测试不同文件类型上传")
    print("=" * 60)

    api_key = os.getenv("GRSAI_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到GRSAI_API_KEY环境变量，跳过此测试")
        return False

    file_types = [
        ("png", "PNG图像"),
        ("jpg", "JPEG图像"),
        ("txt", "文本文件"),
        ("json", "JSON文件"),
    ]

    success_count = 0
    total_tests = len(file_types)

    for ext, description in file_types:
        print(f"\n📝 测试文件类型: {description} (.{ext})")
        print("-" * 40)

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # 根据文件类型创建内容
            if ext in ["png", "jpg"]:
                create_test_image(temp_path)
            elif ext == "txt":
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(
                        f"测试文本文件内容\nFile type: {ext}\nTimestamp: {time.time()}"
                    )
            elif ext == "json":
                import json

                test_data = {"test": True, "file_type": ext, "timestamp": time.time()}
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(test_data, f, indent=2, ensure_ascii=False)

            print(f"📂 文件大小: {os.path.getsize(temp_path)} 字节")

            # 上传文件
            start_time = time.time()
            result_url = upload_file_zh(temp_path)
            end_time = time.time()

            if result_url:
                print(f"✅ {description} 上传成功!")
                print(f"🔗 URL: {result_url}")
                print(f"⏱️ 耗时: {end_time - start_time:.2f}秒")
                success_count += 1
            else:
                print(f"❌ {description} 上传失败")

        except Exception as e:
            print(f"❌ {description} 上传异常: {e}")
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    print(f"\n文件类型测试完成: {success_count}/{total_tests} 通过")
    return success_count == total_tests


def test_get_upload_token_zh():
    """
    测试获取上传token的功能
    """
    print("\n" + "=" * 60)
    print("测试获取上传token功能")
    print("=" * 60)

    api_key = os.getenv("GRSAI_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到GRSAI_API_KEY环境变量，跳过此测试")
        return False

    try:
        print("⏳ 请求上传token...")
        start_time = time.time()

        result = get_upload_token_zh(api_key, {"sux": "png"})

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ 获取token成功!")
        print(f"⏱️ 耗时: {duration:.2f}秒")
        print(f"📝 响应数据结构:")

        if isinstance(result, dict):
            if "data" in result:
                data = result["data"]
                required_fields = ["token", "key", "domain"]
                for field in required_fields:
                    if field in data:
                        value = data[field]
                        if field in ["token", "key"]:
                            # 只显示前10个字符
                            display_value = (
                                f"{value[:10]}..." if len(value) > 10 else value
                            )
                        else:
                            display_value = value
                        print(f"  ✅ {field}: {display_value}")
                    else:
                        print(f"  ❌ 缺少字段: {field}")
            else:
                print("  ❌ 响应中缺少 'data' 字段")
        else:
            print(f"  ❌ 响应不是字典类型: {type(result)}")

        return True

    except Exception as e:
        print(f"❌ 获取token失败: {e}")
        return False


def main():
    """
    主测试函数
    """
    print("🚀 开始测试 upload_file_zh 相关功能")
    print("请确保已设置环境变量 GRSAI_API_KEY")

    all_tests_passed = True

    # 运行各项测试
    tests = [
        ("基本上传功能", test_upload_file_zh_basic),
        ("错误场景", test_upload_file_zh_error_scenarios),
        ("不同文件类型", test_different_file_types),
        ("获取上传token", test_get_upload_token_zh),
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} 测试通过")
                passed_tests += 1
            else:
                print(f"❌ {test_name} 测试失败")
                all_tests_passed = False
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            all_tests_passed = False

    # 总结
    print("\n" + "=" * 80)
    print(f"测试总结: {passed_tests}/{total_tests} 测试通过")

    if all_tests_passed:
        print("🎉 所有测试都通过了!")
    else:
        print("⚠️ 部分测试失败，请检查上述错误信息")

    print("=" * 80)

    return all_tests_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试运行异常: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
