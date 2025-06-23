"""
GrsAI API客户端
封装与grsai.com的所有交互逻辑
"""

import json
import time
import requests
import re
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:
    from PIL import Image

try:
    from .config import GrsaiConfig, default_config
    from .utils import format_error_message, download_image
except ImportError:
    from config import GrsaiConfig, default_config
    from utils import format_error_message, download_image


class GrsaiAPIError(Exception):
    """API调用异常"""

    pass


class GrsaiAPI:
    """GrsAI API客户端类"""

    def __init__(self, api_key: str, config: Optional[GrsaiConfig] = None):
        """
        初始化API客户端

        Args:
            api_key: API密钥
            config: 配置对象
        """
        if not api_key or not api_key.strip():
            raise GrsaiAPIError("API密钥在初始化时不能为空")

        self.api_key = api_key
        self.config = config or default_config
        self.session = requests.Session()
        self._setup_session()

    def _setup_session(self):
        """设置HTTP会话"""
        self.session.headers.update(
            {
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "ComfyUI-GrsAI/1.0",
            }
        )

        # 直接使用传入的API密钥设置认证头
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        发送HTTP请求

        Args:
            method: HTTP方法
            endpoint: API端点
            data: 请求数据
            timeout: 超时时间

        Returns:
            Dict: API响应数据

        Raises:
            GrsaiAPIError: API调用失败
        """
        url = f"{self.config.get_config('api_base_url')}{endpoint}"
        timeout = timeout or self.config.get_config("timeout", 300)
        max_retries = self.config.get_config("max_retries", 3)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        for attempt in range(max_retries):
            try:
                if method.upper() == "POST":
                    response = self.session.post(
                        url, json=data, timeout=timeout, headers=headers
                    )
                else:
                    response = self.session.get(url, timeout=timeout, headers=headers)

                # 检查HTTP状态码
                if response.status_code == 200:
                    json_data = ""
                    if response.text.startswith("data: "):
                        json_data = response.text[6:]
                    else:
                        json_data = response.text
                    # 打印response结果，字符串
                    result = json.loads(json_data)
                    return result
                elif response.status_code == 401:
                    raise GrsaiAPIError("API密钥无效或已过期")
                elif response.status_code == 429:
                    raise GrsaiAPIError("请求频率过高，请稍后重试")
                elif response.status_code >= 500:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)  # 指数退避
                        continue
                    raise GrsaiAPIError(f"服务器错误: {response.status_code}")
                else:
                    error_msg = f"API请求失败: {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg += f" - {error_data['error']}"
                    except:
                        pass
                    raise GrsaiAPIError(error_msg)

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise GrsaiAPIError("请求超时，请检查网络连接")
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise GrsaiAPIError("网络连接失败，请检查网络设置")
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise GrsaiAPIError(format_error_message(e, "网络请求"))

        raise GrsaiAPIError("达到最大重试次数，请求失败")

    def gpt_image_generate_image(
        self,
        prompt: str,
        model: str = "sora-image",
        size: Optional[str] = None,
        urls: List[str] = [],
        variants: Optional[int] = None,
    ) -> Tuple[List["Image.Image"], List[str], List[str]]:
        # 构建请求数据
        payload = {
            "model": model,
            "prompt": prompt,
            "urls": urls,
            "shutProgress": True,
        }

        # 动态添加所有非空的可选参数
        # 这种方式更简洁且易于维护
        optional_params = {
            "size": size,
            "variants": variants,
        }

        for key, value in optional_params.items():
            # 只有当值不是None，或者对于字符串，不是空字符串时，才添加到payload
            if value is not None and value != "":
                payload[key] = value

        # 发送请求
        try:
            response = self._make_request("POST", "/v1/draw/completions", data=payload)
        except Exception as e:
            # 确保将所有底层异常统一包装成我们的自定义异常
            if isinstance(e, GrsaiAPIError):
                raise e
            raise GrsaiAPIError(format_error_message(e, "图像生成"))

        status = response["status"]
        if status != "succeeded":
            raise GrsaiAPIError(f"图像生成失败: {response['id']}")

        results = response["results"]
        resultsUrls = [result["url"] for result in results]
        pil_images = []
        image_urls = []
        errors = []

        def thread_download_image(url):
            try:
                # 下载图像
                print(f"⬇️ 正在下载生成的图像...")
                timeout = self.config.get_config("timeout", 120)
                pil_image = download_image(url, timeout=timeout)
                if pil_image is None:
                    raise GrsaiAPIError(
                        "图像生成成功，但图像下载失败，可能是网络超时或服务异常"
                    )
                print(f"✅ 图像生成并下载成功")
                # 直接返回PIL图像和URL，这是与之前最大的不同
                return pil_image, url
            except Exception as e:
                raise GrsaiAPIError(f"下载或处理图像时出错: {str(e)}")

        with ThreadPoolExecutor(max_workers=len(resultsUrls)) as executor:
            futures = {
                executor.submit(thread_download_image, s): s for s in resultsUrls
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, Exception):
                        # 简化错误信息，不显示技术细节
                        errors.append(f"图像生成失败")
                    else:
                        pil_img, url = result
                        pil_images.append(pil_img)
                        image_urls.append(url)
                except Exception as exc:
                    errors.append(f"图像生成异常")
        return pil_images, image_urls, errors

    def flux_generate_image(
        self,
        prompt: str,
        model: str = "flux-kontext-pro",
        seed: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        urls: List[str] = [],
        output_format: Optional[str] = None,
        safety_tolerance: Optional[int] = None,
        prompt_upsampling: Optional[bool] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Tuple["Image.Image", str]:
        # 构建请求数据
        payload = {
            "model": model,
            "prompt": prompt,
            "urls": urls,
            "shutProgress": True,
        }

        # 动态添加所有非空的可选参数
        # 这种方式更简洁且易于维护
        optional_params = {
            "seed": seed,
            "aspectRatio": aspect_ratio,
            "output_format": output_format,
            "safetyTolerance": safety_tolerance,
            "promptUpsampling": prompt_upsampling,
            "guidance": guidance_scale,
            "steps": num_inference_steps,
        }

        for key, value in optional_params.items():
            # 只有当值不是None，或者对于字符串，不是空字符串时，才添加到payload
            if value is not None and value != "":
                payload[key] = value

        # 发送请求
        try:
            response = self._make_request("POST", "/v1/draw/flux", data=payload)
        except Exception as e:
            # 确保将所有底层异常统一包装成我们的自定义异常
            if isinstance(e, GrsaiAPIError):
                raise e
            raise GrsaiAPIError(format_error_message(e, "图像生成"))

        image_url = response["url"]
        print(image_url)
        if not isinstance(image_url, str) or not image_url.startswith("http"):
            raise GrsaiAPIError(f"API返回了无效的图片URL格式: {str(image_url)[:100]}")

        try:
            # 下载图像
            print("⬇️ 正在下载生成的图像...")
            timeout = self.config.get_config("timeout", 120)  # 提供一个默认值
            pil_image = download_image(image_url, timeout=timeout)
            if pil_image is None:
                # 这里的错误信息可以更具体
                raise GrsaiAPIError("图像下载失败，可能是网络超时或服务异常")

            print("✅ 图像生成并下载成功")
            # 直接返回PIL图像和URL，这是与之前最大的不同
            return pil_image, image_url

        except Exception as e:
            raise GrsaiAPIError(f"下载或处理图像时出错: {str(e)}")

    def test_connection(self) -> bool:
        """
        测试API连接

        Returns:
            bool: 连接是否成功
        """
        try:
            # 尝试一个简单的请求来测试连接
            self.flux_generate_image("test", seed=1)
            return True
        except:
            return False

    def get_api_status(self) -> Dict[str, Any]:
        """
        获取API状态信息

        Returns:
            Dict: 状态信息
        """
        status = {
            "api_key_valid": bool(self.config.get_api_key()),
            "base_url": self.config.get_config("api_base_url"),
            "model": self.config.get_config("model"),
            "timeout": self.config.get_config("timeout"),
            "max_retries": self.config.get_config("max_retries"),
        }

        # 测试连接
        try:
            status["connection_ok"] = self.test_connection()
        except:
            status["connection_ok"] = False

        return status
