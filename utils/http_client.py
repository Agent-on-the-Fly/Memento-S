"""
轻量级异步 HTTP 客户端封装。

统一处理：
- 业务响应结构解析（{"code": int, "data": ..., "msg": str}）
- 422 校验错误提取
- 超时 / 网络异常转换为 (False, message) 元组
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _emit_auth_required() -> None:
    """发布 AUTH_REQUIRED 事件（懒导入避免循环依赖）。"""
    try:
        from utils.event_bus import event_bus, EventType
        event_bus.publish(EventType.AUTH_REQUIRED, source="HttpClient")
    except Exception as e:
        logger.warning("[HttpClient] Failed to emit AUTH_REQUIRED: %s", e)

_DEFAULT_TIMEOUT = 30


@dataclass
class ApiResponse:
    """封装业务层响应结果。"""

    success: bool
    msg: str
    data: Any = field(default=None)
    http_status: int = field(default=0)


class HttpClient:
    """
    无状态的异步 HTTP 请求封装。

    每次请求创建短连接（httpx.AsyncClient），适合低频调用场景。
    如需复用连接，可将 client 提升为实例变量并在外部管理生命周期。
    """

    def __init__(self, base_url: str = "", timeout: float = _DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # 公共请求方法
    # ------------------------------------------------------------------

    async def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> ApiResponse:
        """发送 POST 请求并解析业务响应。"""
        return await self._request("POST", path, json=json, headers=headers)

    async def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> ApiResponse:
        """发送 GET 请求并解析业务响应。"""
        return await self._request("GET", path, params=params, headers=headers)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> ApiResponse:
        url = f"{self.base_url}{path}" if path.startswith("/") else path
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.request(
                    method, url, json=json, params=params, headers=headers
                )
                logger.info("[HttpClient] %s %s → %s", method, url, resp.status_code)
                return self._parse_response(resp)
        except httpx.TimeoutException:
            logger.warning("[HttpClient] %s %s timed out", method, url)
            return ApiResponse(success=False, msg="请求超时，请检查网络连接")
        except httpx.ConnectError as e:
            logger.error("[HttpClient] %s %s connect error: %s", method, url, e)
            return ApiResponse(success=False, msg="无法连接服务器，请检查网络或服务地址")
        except Exception as e:
            logger.error("[HttpClient] %s %s unexpected error: %s", method, url, e)
            return ApiResponse(success=False, msg=str(e))

    @staticmethod
    def _parse_response(resp: httpx.Response) -> ApiResponse:
        """将 HTTP 响应映射为 ApiResponse。"""
        http_status = resp.status_code

        # 认证失效，通知 GUI 弹出登录框
        if http_status == 401:
            _emit_auth_required()
            return ApiResponse(
                success=False,
                msg="登录已过期，请重新登录",
                http_status=http_status,
            )

        # FastAPI 参数校验错误
        if http_status == 422:
            return ApiResponse(
                success=False,
                msg=HttpClient._extract_422_msg(resp),
                http_status=http_status,
            )

        try:
            body = resp.json()
        except Exception:
            return ApiResponse(
                success=False,
                msg=f"响应解析失败（HTTP {http_status}）",
                http_status=http_status,
            )

        biz_code = body.get("code")
        msg = body.get("msg", "")
        data = body.get("data")

        if biz_code == 200:
            return ApiResponse(success=True, msg=msg, data=data, http_status=http_status)

        return ApiResponse(
            success=False,
            msg=msg or f"业务错误（code={biz_code}）",
            data=data,
            http_status=http_status,
        )

    @staticmethod
    def _extract_422_msg(resp: httpx.Response) -> str:
        try:
            detail = resp.json().get("detail", [])
            if detail and isinstance(detail, list):
                return detail[0].get("msg", "参数校验失败")
        except Exception:
            pass
        return "参数校验失败"
