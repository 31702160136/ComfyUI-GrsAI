"""
配置管理模块
处理API密钥、默认参数和环境变量
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

# 尝试导入dotenv，如果失败则优雅降级
try:
    from dotenv import load_dotenv

    # 获取当前插件的根目录，并加载该目录下的.env文件
    module_dir = Path(__file__).parent
    dotenv_path = module_dir / ".env"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"成功从 {dotenv_path} 加载 .env 文件。")
except ImportError:
    print(
        "未找到 python-dotenv 库，将仅从环境变量读取配置。建议安装: pip install python-dotenv"
    )
    load_dotenv = None


class GrsaiConfig:
    """GrsAI配置管理类"""

    # 默认配置
    DEFAULT_CONFIG = {
        "api_base_url": "https://api.grsai.com",
        "model": "flux-kontext-pro",
        "aspect_ratio": "1:1",
        "output_format": "jpeg",
        "safety_tolerance": 2,
        "prompt_upsampling": False,
        "timeout": 300,
        "max_retries": 3,
    }

    # 支持的宽高比
    SUPPORTED_ASPECT_RATIOS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"]

    # 支持的输出格式
    SUPPORTED_OUTPUT_FORMATS = ["jpeg", "png"]

    def __init__(self):
        """
        初始化配置。API密钥将通过get_api_key()方法动态获取。
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.api_key_error_message = self._create_api_key_error_message()

    def _create_api_key_error_message(self) -> str:
        """创建当API密钥未找到时的详细错误消息"""
        module_dir = Path(__file__).parent.resolve()

        return (
            "❌ 未找到API密钥！\n\n"
            "请按以下两种方式之一设置您的GrsAI API密钥：\n\n"
            "方法一 (推荐): 创建 .env 文件\n"
            f"1. 在本插件的根目录中创建一个名为 .env 的文件。\n"
            f"   插件目录: {module_dir}\n"
            "2. 在文件中添加以下内容 (将your-api-key替换为您的真实密钥):\n"
            "   GRSAI_API_KEY=your-api-key-here\n\n"
            "方法二: 设置环境变量\n"
            "1. 设置一个名为 GRSAI_API_KEY 的系统环境变量，值为您的密钥。\n\n"
            "密钥获取地址: https://grsai.com"
        )

    def get_api_key(self) -> Optional[str]:
        """
        从环境变量或.env文件获取API密钥。
        如果找到，返回密钥字符串。
        如果未找到，返回None。
        """
        api_key = os.getenv("GRSAI_API_KEY")
        if api_key and api_key.strip():
            return api_key.strip()
        return None

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any):
        """设置配置项"""
        self.config[key] = value

    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.config.copy()

    def validate_aspect_ratio(self, aspect_ratio: str) -> bool:
        """验证宽高比是否支持"""
        return aspect_ratio in self.SUPPORTED_ASPECT_RATIOS

    def validate_output_format(self, output_format: str) -> bool:
        """验证输出格式是否支持"""
        return output_format in self.SUPPORTED_OUTPUT_FORMATS

    def validate_safety_tolerance(self, safety_tolerance: int) -> bool:
        """验证安全容忍度是否在有效范围内"""
        return 0 <= safety_tolerance <= 6


# 全局配置实例
default_config = GrsaiConfig()
