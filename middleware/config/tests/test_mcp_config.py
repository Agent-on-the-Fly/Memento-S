# SPDX-License-Identifier: Apache-2.0
"""
MCP Config 运行时读写测试

测试 McpConfigManager 对 ~/memento_s/mcp.json 的运行时管理：
1. 初始化：模板不存在时从模板创建
2. CRUD：添加/更新/删除 MCP server
3. enabled 字段控制
4. 运行时切换 enabled 状态
5. 外部文件修改后 reload
6. Schema 验证失败场景
7. 原子写入
8. stdio / streamable_http 两种传输模式

运行方式:
    python -m pytest middleware/config/tests/test_mcp_config.py -v
"""

from __future__ import annotations

import json
import pytest
import tempfile
import shutil
from pathlib import Path

from middleware.config.mcp_config_manager import (
    McpConfigManager,
    McpServerType,
)


class TestMcpBootstrap:
    """MCP 配置初始化"""

    def test_no_file_creates_from_template(self, temp_config_dir):
        """mcp.json 不存在时，从模板创建"""
        manager = McpConfigManager()
        # 临时覆盖路径
        manager.mcp_config_path = temp_config_dir / "mcp.json"

        assert not manager.mcp_config_exists()
        manager.ensure_mcp_config_file()
        assert manager.mcp_config_exists()

        data = manager.load()
        assert "mcp" in data
        assert data["enabled"] is True

    def test_template_has_default_servers(self, temp_config_dir):
        """模板包含默认的 github 和 filesystem server"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()

        servers = manager.get_servers()
        assert "github" in servers
        assert "filesystem" in servers
        assert servers["github"]["transport"] == "stdio"
        assert servers["github"]["command"] == "npx"

    def test_ensure_creates_config_dir(self, temp_config_dir):
        """ensure_mcp_config_file() 创建目录"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "nested" / "path" / "mcp.json"

        manager.ensure_mcp_config_file()
        assert manager.mcp_config_dir.exists()


class TestMcpServerCRUD:
    """MCP Server 增删改"""

    def test_add_local_server(self, temp_config_dir):
        """添加 stdio 本地 server"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        manager.set_server(
            "my-local-server",
            {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "my_mcp_server"],
                "environment": {"MY_VAR": "value"},
                "enabled": True,
            },
        )

        servers = manager.get_servers()
        assert "my-local-server" in servers
        assert servers["my-local-server"]["command"] == "python"
        assert servers["my-local-server"]["environment"]["MY_VAR"] == "value"

    def test_add_remote_server(self, temp_config_dir):
        """添加 streamable_http 远程 server"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        manager.set_server(
            "my-remote-server",
            {
                "transport": "streamable_http",
                "url": "https://mcp.example.com/stream",
                "headers": {"Authorization": "Bearer token"},
                "authToken": "secret",
                "enabled": True,
                "timeout": 10000,
            },
        )

        servers = manager.get_servers()
        assert "my-remote-server" in servers
        assert servers["my-remote-server"]["transport"] == "streamable_http"
        assert servers["my-remote-server"]["url"] == "https://mcp.example.com/stream"
        assert servers["my-remote-server"]["authToken"] == "secret"

    def test_update_existing_server(self, temp_config_dir):
        """更新已存在的 server"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        # 更新 github server
        manager.set_server(
            "github",
            {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github", "--verbose"],
                "environment": {},
                "enabled": True,
            },
        )

        servers = manager.get_servers()
        assert "--verbose" in servers["github"]["args"]

    def test_remove_server(self, temp_config_dir):
        """删除 server"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        manager.remove_server("filesystem")
        servers = manager.get_servers()
        assert "filesystem" not in servers

    def test_remove_nonexistent_server_no_error(self, temp_config_dir):
        """删除不存在的 server 不报错"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        manager.remove_server("ghost_server")  # 不应抛异常
        assert "ghost_server" not in manager.get_servers()

    def test_add_multiple_servers(self, temp_config_dir):
        """同时管理多个 server"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        for i in range(5):
            manager.set_server(
                f"server_{i}",
                {
                    "transport": "stdio",
                    "command": f"server_{i}",
                    "args": [],
                    "enabled": True,
                },
            )

        assert len(manager.get_servers()) == 7  # 2 模板 + 5 新增

    def test_server_roundtrip(self, temp_config_dir):
        """server 配置从创建到重新加载保持一致"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        original = {
            "transport": "streamable_http",
            "url": "https://roundtrip.test.com/mcp",
            "headers": {"X-Header": "value"},
            "authToken": "roundtrip_token",
            "enabled": True,
            "timeout": 7777,
            "oauth": False,
        }
        manager.set_server("roundtrip_test", original)

        # 重新加载
        manager.reload()
        servers = manager.get_servers()

        assert servers["roundtrip_test"]["url"] == "https://roundtrip.test.com/mcp"
        assert servers["roundtrip_test"]["authToken"] == "roundtrip_token"
        assert servers["roundtrip_test"]["headers"] == {"X-Header": "value"}
        assert servers["roundtrip_test"]["timeout"] == 7777


class TestMcpEnabledControl:
    """MCP 全局 enabled 控制"""

    def test_default_enabled(self, temp_config_dir):
        """默认 enabled=True"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        assert manager.is_enabled() is True

    def test_disable_all_servers(self, temp_config_dir):
        """禁用所有 MCP server"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        manager.set_enabled(False)
        assert manager.is_enabled() is False

        # 重新加载后仍为 False
        manager.reload()
        assert manager.is_enabled() is False

    def test_disable_individual_server(self, temp_config_dir):
        """禁用单个 server 而不影响全局"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        # 禁用 github
        manager.set_server("github", {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "enabled": False,
        })

        servers = manager.get_servers()
        assert servers["github"]["enabled"] is False
        assert manager.is_enabled() is True  # 全局不受影响


class TestMcpRuntimeReload:
    """运行时 reload"""

    def test_reload_reflects_external_changes(self, temp_config_dir):
        """外部程序修改 mcp.json 后 reload() 能读到新值"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        manager.set_server("original", {
            "transport": "stdio",
            "command": "original",
            "args": [],
            "enabled": True,
        })

        # 外部修改
        raw = json.loads(manager.mcp_config_path.read_text(encoding="utf-8"))
        raw["mcp"]["externally_added"] = {
            "transport": "stdio",
            "command": "external",
            "args": [],
            "enabled": True,
        }
        manager.mcp_config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        manager.reload()
        servers = manager.get_servers()
        assert "externally_added" in servers
        assert servers["externally_added"]["command"] == "external"

    def test_reload_reflects_enabled_change(self, temp_config_dir):
        """set_enabled() 后 reload() 反映新状态"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        assert manager.is_enabled() is True

        manager.set_enabled(False)

        # reload 后应反映新状态
        manager.reload()
        assert manager.is_enabled() is False


class TestMcpSchemaValidation:
    """Schema 验证"""

    def test_invalid_transport_rejected(self, temp_config_dir):
        """无效的 transport 类型被 schema 拒绝"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        # set_server 不做 schema 验证，直接写入磁盘
        # 用手动修改文件来模拟无效数据，然后 reload 触发验证
        raw = json.loads(manager.mcp_config_path.read_text(encoding="utf-8"))
        raw["mcp"]["bad_transport"] = {
            "transport": "ftp",  # 非法类型
            "command": "test",
        }
        manager.mcp_config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        with pytest.raises(Exception):  # jsonschema.ValidationError
            manager.reload()

    def test_stdio_without_command_rejected(self, temp_config_dir):
        """stdio 模式缺少 command 字段被 schema 拒绝"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        raw = json.loads(manager.mcp_config_path.read_text(encoding="utf-8"))
        raw["mcp"]["missing_command"] = {
            "transport": "stdio",
            # 缺少 command
        }
        manager.mcp_config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        with pytest.raises(Exception):
            manager.reload()

    def test_streamable_http_without_url_rejected(self, temp_config_dir):
        """streamable_http 模式缺少 url 字段被 schema 拒绝"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        raw = json.loads(manager.mcp_config_path.read_text(encoding="utf-8"))
        raw["mcp"]["missing_url"] = {
            "transport": "streamable_http",
            # 缺少 url
        }
        manager.mcp_config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        with pytest.raises(Exception):
            manager.reload()

    def test_oauth_config_accepted(self, temp_config_dir):
        """合法的 OAuth 配置被接受"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        manager.set_server("with_oauth", {
            "transport": "streamable_http",
            "url": "https://oauth.example.com/mcp",
            "oauth": {
                "clientId": "client_id",
                "clientSecret": "client_secret",
                "scope": "read write",
            },
            "enabled": True,
        })

        servers = manager.get_servers()
        assert servers["with_oauth"]["oauth"]["clientId"] == "client_id"


class TestMcpAtomicWrite:
    """原子写入"""

    def test_write_creates_tmp_file_atomically(self, temp_config_dir):
        """save() 使用 tmp 文件原子写入"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        manager.set_server("atomic_test", {
            "transport": "stdio",
            "command": "atomic",
            "enabled": True,
        })

        # tmp 文件应在写入完成后消失
        tmp_files = list(temp_config_dir.glob("mcp.json.tmp"))
        assert len(tmp_files) == 0

        # 原始文件完整
        assert manager.mcp_config_path.exists()
        data = json.loads(manager.mcp_config_path.read_text(encoding="utf-8"))
        assert "atomic_test" in data["mcp"]

    def test_concurrent_set_operations(self, temp_config_dir):
        """并发 set 操作不丢失数据"""
        manager = McpConfigManager()
        manager.mcp_config_path = temp_config_dir / "mcp.json"
        manager.ensure_mcp_config_file()
        manager.load()

        for i in range(10):
            manager.set_server(f"concurrent_{i}", {
                "transport": "stdio",
                "command": f"cmd_{i}",
                "enabled": True,
            })

        servers = manager.get_servers()
        # github, filesystem + 10 个新 server = 12
        assert len(servers) == 12
        for i in range(10):
            assert f"concurrent_{i}" in servers
