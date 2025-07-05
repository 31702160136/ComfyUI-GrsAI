#!/usr/bin/env python3


def test_flux_nodes():
    try:
        from flux_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        print("=== 测试 Flux 节点 ===")
        print(f"找到 {len(NODE_CLASS_MAPPINGS)} 个Flux节点类")

        for node_name, node_class in NODE_CLASS_MAPPINGS.items():
            print(f"\n测试节点: {node_name}")
            print(f"  类名: {node_class.__name__}")
            print(f"  显示名: {NODE_DISPLAY_NAME_MAPPINGS.get(node_name, 'N/A')}")

            # 检查必须的类方法
            required_methods = ["INPUT_TYPES"]
            for method in required_methods:
                if hasattr(node_class, method):
                    print(f"  ✅ 有 {method} 方法")
                    try:
                        # 尝试调用INPUT_TYPES
                        if method == "INPUT_TYPES":
                            input_types = node_class.INPUT_TYPES()
                            print(f"    INPUT_TYPES 返回类型: {type(input_types)}")
                            if isinstance(input_types, dict):
                                print(
                                    f"    required 参数数量: {len(input_types.get('required', {}))}"
                                )
                                print(
                                    f"    optional 参数数量: {len(input_types.get('optional', {}))}"
                                )
                            else:
                                print(
                                    f"    ❌ INPUT_TYPES 返回了错误类型: {type(input_types)}"
                                )
                    except Exception as e:
                        print(f"    ❌ 调用 {method} 失败: {e}")
                else:
                    print(f"  ❌ 缺少 {method} 方法")

            # 检查必须的类属性
            required_attrs = ["RETURN_TYPES", "RETURN_NAMES", "FUNCTION", "CATEGORY"]
            for attr in required_attrs:
                if hasattr(node_class, attr):
                    value = getattr(node_class, attr)
                    print(f"  ✅ 有 {attr}: {value}")
                else:
                    print(f"  ❌ 缺少 {attr} 属性")

            # 尝试实例化
            try:
                instance = node_class()
                print(f"  ✅ 可以实例化")

                # 检查execute方法
                if hasattr(instance, "execute"):
                    print(f"  ✅ 有 execute 方法")
                else:
                    print(f"  ❌ 缺少 execute 方法")

            except Exception as e:
                print(f"  ❌ 实例化失败: {e}")

    except Exception as e:
        print(f"❌ 测试Flux节点失败: {e}")
        import traceback

        traceback.print_exc()


def test_gpt_nodes():
    try:
        from gpt_image_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        print("\n=== 测试 GPT Image 节点 ===")
        print(f"找到 {len(NODE_CLASS_MAPPINGS)} 个GPT节点类")

        for node_name, node_class in NODE_CLASS_MAPPINGS.items():
            print(f"\n测试节点: {node_name}")
            print(f"  类名: {node_class.__name__}")
            print(f"  显示名: {NODE_DISPLAY_NAME_MAPPINGS.get(node_name, 'N/A')}")

            # 只做基本检查，不需要重复详细测试
            try:
                instance = node_class()
                input_types = node_class.INPUT_TYPES()
                print(f"  ✅ 基本检查通过")
            except Exception as e:
                print(f"  ❌ 基本检查失败: {e}")

    except Exception as e:
        print(f"❌ 测试GPT节点失败: {e}")


if __name__ == "__main__":
    test_flux_nodes()
    test_gpt_nodes()
