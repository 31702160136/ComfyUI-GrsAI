"""ComfyUI-GrsAI 的统一 HTTP 会话与网络错误处理。"""

import ssl
from typing import Any, Dict, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import truststore
except ImportError:
    truststore = None


TimeoutValue = Union[int, float, Tuple[Union[int, float], Union[int, float]]]


def normalize_timeout(
    timeout: TimeoutValue,
    default_connect_timeout: Union[int, float] = 15,
) -> Tuple[float, float]:
    """将单个超时值转换为明确的（连接超时，读取超时）。"""
    if isinstance(timeout, tuple):
        return float(timeout[0]), float(timeout[1])

    read_timeout = max(1.0, float(timeout))
    connect_timeout = min(float(default_connect_timeout), read_timeout)
    return connect_timeout, read_timeout


def _create_system_ssl_context() -> Optional[ssl.SSLContext]:
    """优先使用操作系统证书库；不可用时由 Requests 使用 certifi。"""
    if truststore is None:
        return None
    try:
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception as exc:
        print(f"⚠️ 系统证书库初始化失败，将使用 Python 证书库: {exc}")
        return None


class _SystemTrustHTTPAdapter(HTTPAdapter):
    """让当前插件的请求使用系统证书库，不影响 ComfyUI 中的其他插件。"""

    def __init__(
        self,
        *args: Any,
        ssl_context: Optional[ssl.SSLContext] = None,
        **kwargs: Any,
    ) -> None:
        self._ssl_context = ssl_context
        super().__init__(*args, **kwargs)

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = False,
        **pool_kwargs: Any,
    ) -> None:
        if self._ssl_context is not None:
            pool_kwargs["ssl_context"] = self._ssl_context
        super().init_poolmanager(connections, maxsize, block, **pool_kwargs)

    def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any):
        if self._ssl_context is not None:
            proxy_kwargs.setdefault("ssl_context", self._ssl_context)
        return super().proxy_manager_for(proxy, **proxy_kwargs)

    # Requests 2.32+ 会按请求创建 TLS 连接池，需要在这里保留系统 SSLContext。
    def build_connection_pool_key_attributes(
        self,
        request: requests.PreparedRequest,
        verify: Any,
        cert: Any = None,
    ):
        parent_method = getattr(
            super(), "build_connection_pool_key_attributes", None
        )
        if parent_method is None:
            raise AttributeError("当前 Requests 版本不支持连接池 TLS 属性")

        host_params, pool_kwargs = parent_method(request, verify, cert)
        if self._ssl_context is not None and verify is not False:
            pool_kwargs["ssl_context"] = self._ssl_context
            pool_kwargs.pop("ca_certs", None)
            pool_kwargs.pop("ca_cert_dir", None)
        return host_params, pool_kwargs


def create_http_session(
    headers: Optional[Dict[str, str]] = None,
    retries: int = 3,
    pool_maxsize: int = 16,
) -> requests.Session:
    """创建带安全系统证书、连接池和幂等请求重试的会话。"""
    retry_policy = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
        status_forcelist=(408, 425, 429, 500, 502, 503, 504),
        backoff_factor=0.75,
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    ssl_context = _create_system_ssl_context()
    adapter_kwargs = {
        "max_retries": retry_policy,
        "pool_connections": pool_maxsize,
        "pool_maxsize": pool_maxsize,
    }

    session = requests.Session()
    # 保持默认 True，以便自动读取用户的 HTTP(S)_PROXY / NO_PROXY 配置。
    session.trust_env = True
    session.mount("http://", HTTPAdapter(**adapter_kwargs))
    session.mount(
        "https://",
        _SystemTrustHTTPAdapter(
            ssl_context=ssl_context,
            **adapter_kwargs,
        ),
    )
    if headers:
        session.headers.update(headers)
    return session


def describe_request_error(
    exc: requests.RequestException,
    action: str = "网络请求",
) -> str:
    """把 Requests 异常转换成普通用户可理解、可排查的信息。"""
    if isinstance(exc, requests.exceptions.SSLError):
        return (
            f"{action}失败：TLS 证书校验失败。"
            "插件已尝试使用系统证书库，请确认系统时间正确并更新插件依赖"
        )
    if isinstance(exc, requests.exceptions.ProxyError):
        return (
            f"{action}失败：无法连接网络代理。"
            "请检查代理软件是否运行，或将结果文件域名设置为直连"
        )
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return f"{action}失败：连接服务器超时，请检查代理、DNS 或防火墙设置"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return f"{action}失败：服务器响应超时，插件重试后仍未成功"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"{action}失败：无法连接服务器，请检查代理、DNS 或防火墙设置"
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else "未知"
        return f"{action}失败：服务器返回 HTTP {status}"

    detail = str(exc).strip()
    if len(detail) > 300:
        detail = detail[:297] + "..."
    return f"{action}失败：{type(exc).__name__}: {detail or '未知网络错误'}"
