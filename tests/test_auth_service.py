"""
AuthService 单元测试

覆盖场景：
- 发送验证码（成功 / 失败）
- 登录（成功 / 验证码错误 / 网络超时）
- 登出（成功 / 服务端失败）
- 401 强制清除 token（clear_token）
- token 持久化（写入 / 读取 / 清除）
- HttpClient 实例缓存（不重复创建）
- base_url 变更时自动重建 HttpClient

使用方法：
    pytest tests/test_auth_service.py -v
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ── 在导入被测模块前 mock 掉外部依赖 ─────────────────────────────────────

@dataclass
class _MockAuthConfig:
    base_url: str = "https://api.example.com"


@dataclass
class _MockConfig:
    auth: _MockAuthConfig = None

    def __post_init__(self):
        if self.auth is None:
            self.auth = _MockAuthConfig()


_mock_g_config = _MockConfig()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

with patch.dict(
    "sys.modules",
    {
        "middleware.config": MagicMock(g_config=_mock_g_config),
        "utils.path_manager": MagicMock(),
    },
):
    from utils.http_client import ApiResponse
    from gui.modules.auth_service import AuthService


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_token_path(tmp_path: Path):
    """返回一个临时 token 文件路径，并 patch PathManager。"""
    token_file = tmp_path / "auth_token.json"
    return token_file


@pytest.fixture
def service(tmp_token_path: Path):
    """创建一个干净的 AuthService，token 文件指向临时目录。"""
    with patch.object(
        AuthService, "_token_path", return_value=tmp_token_path
    ):
        svc = AuthService()
    return svc


def _make_http_mock(service: AuthService, response: ApiResponse) -> MagicMock:
    """替换 service._http_client，让 post/get 返回指定 ApiResponse。"""
    mock_client = MagicMock()
    mock_client.base_url = service.base_url
    mock_client.post = AsyncMock(return_value=response)
    mock_client.get = AsyncMock(return_value=response)
    service._http_client = mock_client
    return mock_client


# ── 初始状态 ──────────────────────────────────────────────────────────────


class TestInitialState:
    def test_not_logged_in_by_default(self, service: AuthService):
        assert service.is_logged_in is False

    def test_token_is_none_by_default(self, service: AuthService):
        assert service.token is None

    def test_user_info_is_empty_by_default(self, service: AuthService):
        assert service.user_info == {}

    def test_display_name_is_empty_by_default(self, service: AuthService):
        assert service.display_name == ""

    def test_base_url_from_config(self, service: AuthService):
        assert service.base_url == "https://api.example.com"


# ── HttpClient 缓存 ───────────────────────────────────────────────────────


class TestHttpClientCache:
    def test_http_client_created_on_first_access(self, service: AuthService):
        assert service._http_client is None
        client = service._http
        assert client is not None
        assert service._http_client is client

    def test_http_client_reused_on_second_access(self, service: AuthService):
        client1 = service._http
        client2 = service._http
        assert client1 is client2

    def test_http_client_recreated_on_base_url_change(self, service: AuthService):
        client1 = service._http
        # 模拟 base_url 变更
        _mock_g_config.auth.base_url = "https://new.example.com"
        try:
            client2 = service._http
            assert client1 is not client2
        finally:
            _mock_g_config.auth.base_url = "https://api.example.com"


# ── 发送验证码 ────────────────────────────────────────────────────────────


class TestSendVerificationCode:
    @pytest.mark.asyncio
    async def test_send_code_success(self, service: AuthService):
        _make_http_mock(service, ApiResponse(success=True, msg="验证码已发送"))

        ok, msg = await service.send_verification_code("user@example.com")

        assert ok is True
        assert msg == "验证码已发送"
        service._http_client.post.assert_called_once_with(
            "/api/v1/auth/send_verification_code",
            json={"account": "user@example.com"},
        )

    @pytest.mark.asyncio
    async def test_send_code_account_not_found(self, service: AuthService):
        _make_http_mock(service, ApiResponse(success=False, msg="账号不存在"))

        ok, msg = await service.send_verification_code("ghost@example.com")

        assert ok is False
        assert "账号" in msg

    @pytest.mark.asyncio
    async def test_send_code_network_timeout(self, service: AuthService):
        _make_http_mock(
            service, ApiResponse(success=False, msg="请求超时，请检查网络连接")
        )

        ok, msg = await service.send_verification_code("user@example.com")

        assert ok is False
        assert "超时" in msg


# ── 登录 ──────────────────────────────────────────────────────────────────


class TestLogin:
    @pytest.mark.asyncio
    async def test_login_success_stores_token(self, service: AuthService, tmp_token_path: Path):
        _make_http_mock(
            service,
            ApiResponse(
                success=True,
                msg="登录成功",
                data={
                    "token": "tok_abc123",
                    "token_type": "Bearer",
                    "user_info": {"uid": "u001", "nickname": "Alice"},
                },
            ),
        )

        with patch.object(service, "_token_path", return_value=tmp_token_path):
            ok, msg = await service.login("user@example.com", "123456")

        assert ok is True
        assert service.is_logged_in is True
        assert service.token == "tok_abc123"
        assert service.user_info["uid"] == "u001"
        assert service.display_name == "Alice"

    @pytest.mark.asyncio
    async def test_login_success_persists_token(self, service: AuthService, tmp_token_path: Path):
        _make_http_mock(
            service,
            ApiResponse(
                success=True,
                msg="登录成功",
                data={"token": "tok_persist", "token_type": "Bearer", "user_info": {}},
            ),
        )

        with patch.object(service, "_token_path", return_value=tmp_token_path):
            await service.login("user@example.com", "123456")

        assert tmp_token_path.exists()
        saved = json.loads(tmp_token_path.read_text(encoding="utf-8"))
        assert saved["token"] == "tok_persist"

    @pytest.mark.asyncio
    async def test_login_wrong_code_does_not_store_token(self, service: AuthService):
        _make_http_mock(service, ApiResponse(success=False, msg="验证码错误"))

        ok, msg = await service.login("user@example.com", "000000")

        assert ok is False
        assert service.is_logged_in is False
        assert service.token is None

    @pytest.mark.asyncio
    async def test_login_passes_correct_payload(self, service: AuthService):
        _make_http_mock(
            service,
            ApiResponse(
                success=True,
                msg="ok",
                data={"token": "t", "token_type": "Bearer", "user_info": {}},
            ),
        )

        with patch.object(service, "_token_path", return_value=Path("/dev/null")):
            await service.login("hello@x.com", "654321")

        service._http_client.post.assert_called_once_with(
            "/api/v1/auth/login",
            json={"account": "hello@x.com", "code": "654321"},
        )

    @pytest.mark.asyncio
    async def test_login_network_error_returns_false(self, service: AuthService):
        _make_http_mock(
            service, ApiResponse(success=False, msg="无法连接服务器，请检查网络或服务地址")
        )

        ok, msg = await service.login("user@example.com", "123456")

        assert ok is False
        assert service.token is None


# ── 登出 ──────────────────────────────────────────────────────────────────


class TestLogout:
    @pytest.mark.asyncio
    async def test_logout_calls_api_with_token_header(self, service: AuthService):
        service._token = "tok_existing"
        service._token_type = "Bearer"
        _make_http_mock(service, ApiResponse(success=True, msg=""))

        with patch.object(service, "_token_path", return_value=Path("/dev/null")):
            ok, msg = await service.logout()

        assert ok is True
        service._http_client.post.assert_called_once_with(
            "/api/v1/auth/logout",
            headers={"authorization": "Bearer tok_existing"},
        )

    @pytest.mark.asyncio
    async def test_logout_clears_local_state(self, service: AuthService, tmp_token_path: Path):
        service._token = "tok_existing"
        service._user_info = {"uid": "u001"}
        _make_http_mock(service, ApiResponse(success=True, msg=""))
        tmp_token_path.write_text("{}", encoding="utf-8")

        with patch.object(service, "_token_path", return_value=tmp_token_path):
            await service.logout()

        assert service.is_logged_in is False
        assert service.token is None
        assert service.user_info == {}
        assert not tmp_token_path.exists()

    @pytest.mark.asyncio
    async def test_logout_without_token_skips_api_call(self, service: AuthService):
        mock_client = MagicMock()
        mock_client.base_url = service.base_url
        mock_client.post = AsyncMock()
        service._http_client = mock_client

        with patch.object(service, "_token_path", return_value=Path("/dev/null")):
            ok, msg = await service.logout()

        assert ok is True
        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_logout_api_failure_returns_false(self, service: AuthService):
        service._token = "tok_bad"
        _make_http_mock(service, ApiResponse(success=False, msg="服务端异常"))

        ok, msg = await service.logout()

        assert ok is False
        assert "服务端" in msg
        # 服务端拒绝时 token 不应被清除
        assert service.token == "tok_bad"


# ── 401 强制清除 token ────────────────────────────────────────────────────


class TestClearToken:
    def test_clear_token_resets_state(self, service: AuthService, tmp_token_path: Path):
        service._token = "tok_old"
        service._user_info = {"uid": "u001"}
        tmp_token_path.write_text("{}", encoding="utf-8")

        with patch.object(service, "_token_path", return_value=tmp_token_path):
            service.clear_token()

        assert service.is_logged_in is False
        assert service.token is None
        assert service.user_info == {}

    def test_clear_token_removes_token_file(self, service: AuthService, tmp_token_path: Path):
        service._token = "tok_old"
        tmp_token_path.write_text('{"token": "tok_old"}', encoding="utf-8")

        with patch.object(service, "_token_path", return_value=tmp_token_path):
            service.clear_token()

        assert not tmp_token_path.exists()

    def test_clear_token_safe_when_no_file(self, service: AuthService, tmp_token_path: Path):
        """token 文件不存在时 clear_token 不应抛异常。"""
        assert not tmp_token_path.exists()
        with patch.object(service, "_token_path", return_value=tmp_token_path):
            service.clear_token()  # 不抛异常即通过


# ── token 持久化 ──────────────────────────────────────────────────────────


class TestTokenPersistence:
    def test_persisted_token_loaded_on_init(self, tmp_token_path: Path):
        """已有 token 文件时，__init__ 应自动加载。"""
        tmp_token_path.write_text(
            json.dumps(
                {
                    "token": "tok_restored",
                    "token_type": "Bearer",
                    "user_info": {"uid": "u999"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        with patch.object(AuthService, "_token_path", return_value=tmp_token_path):
            svc = AuthService()

        assert svc.is_logged_in is True
        assert svc.token == "tok_restored"
        assert svc.user_info["uid"] == "u999"

    def test_corrupted_token_file_does_not_crash(self, tmp_token_path: Path):
        """损坏的 token 文件不应导致崩溃。"""
        tmp_token_path.write_text("not-valid-json", encoding="utf-8")

        with patch.object(AuthService, "_token_path", return_value=tmp_token_path):
            svc = AuthService()

        assert svc.is_logged_in is False

    def test_display_name_prefers_nickname(self, tmp_token_path: Path):
        tmp_token_path.write_text(
            json.dumps(
                {
                    "token": "t",
                    "token_type": "Bearer",
                    "user_info": {
                        "nickname": "Bob",
                        "email": "bob@x.com",
                        "uid": "u001",
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch.object(AuthService, "_token_path", return_value=tmp_token_path):
            svc = AuthService()

        assert svc.display_name == "Bob"

    def test_display_name_falls_back_to_email(self, tmp_token_path: Path):
        tmp_token_path.write_text(
            json.dumps(
                {
                    "token": "t",
                    "token_type": "Bearer",
                    "user_info": {"email": "carol@x.com", "uid": "u002"},
                }
            ),
            encoding="utf-8",
        )

        with patch.object(AuthService, "_token_path", return_value=tmp_token_path):
            svc = AuthService()

        assert svc.display_name == "carol@x.com"


# ── 401 → AUTH_REQUIRED 事件链路 ─────────────────────────────────────────


class TestAuthRequiredEvent:
    def test_401_response_publishes_auth_required(self):
        """HTTP 401 响应应触发 event_bus 发布 AUTH_REQUIRED 事件。"""
        from utils.event_bus import event_bus, EventType
        from utils.http_client import HttpClient

        received = []
        handler = lambda e: received.append(e)
        event_bus.subscribe(EventType.AUTH_REQUIRED, handler)

        try:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            result = HttpClient._parse_response(mock_resp)
        finally:
            event_bus.unsubscribe(EventType.AUTH_REQUIRED, handler)

        assert len(received) == 1
        assert received[0].source == "HttpClient"
        assert result.success is False
        assert "登录" in result.msg

    def test_non_401_does_not_publish_auth_required(self):
        """非 401 响应不应触发 AUTH_REQUIRED 事件。"""
        from utils.event_bus import event_bus, EventType
        from utils.http_client import HttpClient

        received = []
        handler = lambda e: received.append(e)
        event_bus.subscribe(EventType.AUTH_REQUIRED, handler)

        try:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"code": 200, "data": None, "msg": "ok"}
            HttpClient._parse_response(mock_resp)
        finally:
            event_bus.unsubscribe(EventType.AUTH_REQUIRED, handler)

        assert len(received) == 0

    def test_401_response_source_is_http_client(self):
        """AUTH_REQUIRED 事件的 source 字段应标注来自 HttpClient。"""
        from utils.event_bus import event_bus, EventType
        from utils.http_client import HttpClient

        received = []
        handler = lambda e: received.append(e)
        event_bus.subscribe(EventType.AUTH_REQUIRED, handler)

        try:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            HttpClient._parse_response(mock_resp)
        finally:
            event_bus.unsubscribe(EventType.AUTH_REQUIRED, handler)

        assert received[0].source == "HttpClient"
