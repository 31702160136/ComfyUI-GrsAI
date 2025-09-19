import requests
import json
import os
from typing import Optional, Dict, Any

# Prefer relative import in package context, fallback to absolute for scripts
try:
    from .config import default_config
except ImportError:
    from config import default_config


def get_upload_token(
    api_key: str, data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    向 grsai API 发送 POST 请求获取上传 token

    Args:
        api_key (str): API 密钥
        data (Optional[Dict[str, Any]]): 可选的请求数据

    Returns:
        Dict[str, Any]: API 响应数据

    Raises:
        requests.exceptions.RequestException: 请求异常
        ValueError: 响应数据解析异常
    """
    url = "https://grsai.dakka.com.cn/client/resource/newUploadToken"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        # 发送 POST 请求
        response = requests.post(
            url=url, headers=headers, json=data if data else {}, timeout=30
        )

        # 检查响应状态
        response.raise_for_status()

        # 返回 JSON 响应
        return response.json()

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"请求失败: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"响应数据解析失败: {e}")


def get_upload_token_zh(
    api_key: str, data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    向 grsai API 发送 POST 请求获取上传 token

    Args:
        api_key (str): API 密钥
        data (Optional[Dict[str, Any]]): 可选的请求数据

    Returns:
        Dict[str, Any]: API 响应数据

    Raises:
        requests.exceptions.RequestException: 请求异常
        ValueError: 响应数据解析异常
    """
    url = "https://grsai.dakka.com.cn/client/resource/newUploadTokenZH"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        # 发送 POST 请求
        response = requests.post(
            url=url, headers=headers, json=data if data else {}, timeout=30
        )

        # 检查响应状态
        response.raise_for_status()

        # 返回 JSON 响应
        return response.json()

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"请求失败: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"响应数据解析失败: {e}")


def upload_file(api_key: Optional[str] = None, file_path: str = "") -> str:
    # Backward compatibility: allow single positional file_path as first arg
    if not file_path and isinstance(api_key, str) and api_key and not api_key.startswith("sk-"):
        file_path, api_key = api_key, None
    if not file_path:
        return ""
    """
    上传文件到 R2 存储服务（直传 PUT 方式）。

    Args:
        api_key (Optional[str]): 可选的 API 密钥；如未提供，将回退使用 default_config.get_api_key()。
        file_path (str): 要上传的文件路径。

    Returns:
        str: 文件访问 URL。

    Raises:
        ValueError: 当无法解析有效的 API 密钥时抛出。
        FileNotFoundError: 文件不存在。
        requests.exceptions.RequestException: 上传失败。
    """
    # 解析 API 密钥：优先使用显式参数，否则回退配置
    effective_key = (api_key or "").strip()
    if not effective_key:
        effective_key = default_config.get_api_key() or ""
    if not effective_key or not effective_key.startswith("sk-"):
        # 复用统一错误提示
        raise ValueError(default_config.api_key_error_message)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 从文件路径获取文件扩展名
    file_extension = os.path.splitext(file_path)[1].lstrip(".")
    if not file_extension:
        file_extension = "png"  # 默认扩展名

    # 获取上传 token
    result = get_upload_token(effective_key, {"sux": file_extension})
    url = result["data"]["url"]
    key = result["data"]["key"]
    domain = result["data"]["domain"]

    # 执行文件上传
    try:
        with open(file_path, "rb") as file:
            # 使用预签名 URL 上传文件
            upload_response = requests.put(
                url=url,
                data=file,
                headers={
                    "Content-Type": "application/octet-stream",
                },
                timeout=120,  # 增加超时时间以适应大文件上传
            )

            # 检查上传是否成功
            upload_response.raise_for_status()

            # 返回文件访问URL
            file_url = domain + "/" + key
            return file_url

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"文件上传失败: {e}")
    except IOError as e:
        raise IOError(f"文件读取失败: {e}")


def upload_file_zh(api_key: Optional[str] = None, file_path: str = "") -> str:
    # Backward compatibility: allow single positional file_path as first arg
    if not file_path and isinstance(api_key, str) and api_key and not api_key.startswith("sk-"):
        file_path, api_key = api_key, None
    if not file_path:
        return ""
    """
    上传文件到 R2 存储服务（表单 POST 方式，国内加速域）。

    Args:
        api_key (Optional[str]): 可选的 API 密钥；如未提供，将回退使用 default_config.get_api_key()。
        file_path (str): 要上传的文件路径。

    Returns:
        str: 文件访问 URL。

    Raises:
        ValueError: 当无法解析有效的 API 密钥时抛出。
        FileNotFoundError: 文件不存在。
        requests.exceptions.RequestException: 上传失败。
    """
    # 解析 API 密钥：优先使用显式参数，否则回退配置
    effective_key = (api_key or "").strip()
    if not effective_key:
        effective_key = default_config.get_api_key() or ""
    if not effective_key or not effective_key.startswith("sk-"):
        raise ValueError(default_config.api_key_error_message)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 从文件路径获取文件扩展名
    file_extension = os.path.splitext(file_path)[1].lstrip(".")
    if not file_extension:
        file_extension = "png"  # 默认扩展名

    # 获取上传 token
    result = get_upload_token_zh(effective_key, {"sux": file_extension})
    token = result["data"]["token"]
    key = result["data"]["key"]
    url = result["data"]["url"]
    domain = result["data"]["domain"]

    # 执行文件上传
    try:
        with open(file_path, "rb") as file:
            # 使用预签名 URL 上传文件 (multipart/form-data)
            upload_response = requests.post(
                url=url,
                data={
                    "token": token,
                    "key": key,
                },
                files={
                    "file": file,
                },
                # 不要手动设置 Content-Type，让 requests 自动处理 multipart/form-data 和 boundary
                timeout=120,  # 增加超时时间以适应大文件上传
            )

            # 检查上传是否成功
            upload_response.raise_for_status()

            # 返回文件访问URL
            file_url = domain + "/" + key
            return file_url

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"文件上传失败: {e}")
    except IOError as e:
        raise IOError(f"文件读取失败: {e}")


# 使用示例
if __name__ == "__main__":
    try:
        # 使用默认配置中的 API 密钥，需要提供实际的文件路径
        result = upload_file_zh(file_path="text-to-image-demo.png")
        print("文件上传成功:")
        print(result)

        # 或者使用自定义 API 密钥（显式传参优先于默认配置）
        # result = upload_file(file_path="path/to/your/file.jpg", api_key="sk-xxxxx")
        # print(result)

    except Exception as e:
        print(f"错误: {e}")
