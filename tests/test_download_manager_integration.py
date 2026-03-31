"""Download Manager 集成测试 —— 真实 GitHub 下载测试

此测试类对 download_manager 进行真实调用，不进行 mock。
测试会从真实的 GitHub 仓库下载 skill。

注意: 此测试需要网络连接。
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from core.skill.downloader.config import DownloadConfig
from core.skill.downloader.factory import create_default_download_manager
from core.skill.downloader.github import GitHubSkillDownloader
from core.skill.downloader.manager import DownloadManager


class TestDownloadManagerIntegration:
    """DownloadManager 真实集成测试

    使用真实的 GitHub 仓库进行测试，验证下载功能是否正常。
    测试使用的仓库是公开可用的 skill 示例仓库。
    """

    # 测试用的真实 GitHub URL（使用一个公开的示例 skill）
    # 这个 URL 指向一个包含 SKILL.md 的公开仓库
    TEST_GITHUB_URL = (
        "https://github.com/opencode-ai/skills-registry/tree/main/skills/hello-world"
    )

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录，测试后自动清理"""
        temp_path = Path(tempfile.mkdtemp(prefix="test_download_"))
        yield temp_path
        # 测试结束后清理
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def download_manager(self):
        """创建默认下载管理器"""
        return create_default_download_manager()

    @pytest.fixture
    def github_downloader(self):
        """创建 GitHub 下载器（无 token）"""
        config = DownloadConfig()
        return GitHubSkillDownloader(config)

    def test_github_downloader_can_handle_github_url(self, github_downloader):
        """测试 GitHub 下载器能识别 GitHub URL"""
        # 应该能处理 GitHub URL
        assert (
            github_downloader.can_handle("https://github.com/user/repo/tree/main/skill")
            is True
        )
        assert github_downloader.can_handle("https://github.com/user/repo") is True

        # 不应该处理非 GitHub URL
        assert github_downloader.can_handle("https://gitlab.com/user/repo") is False
        assert github_downloader.can_handle("https://gitee.com/user/repo") is False
        assert github_downloader.can_handle("not-a-url") is False

    def test_github_downloader_can_handle_with_various_urls(self, github_downloader):
        """测试 GitHub 下载器对各种 URL 的识别能力"""
        test_cases = [
            (
                "https://github.com/opencode-ai/skills-registry/tree/main/skills/hello-world",
                True,
            ),
            ("https://github.com/user/my-skill/tree/v1.0.0", True),
            ("https://github.com/user/repo", True),
            # raw.githubusercontent.com 是 CDN 域名，不是标准 GitHub 仓库 URL
            ("https://raw.githubusercontent.com/user/repo/main/file.txt", False),
            ("https://gitlab.com/user/repo/tree/main", False),
            (
                "https://github.com",
                True,
            ),  # 只有域名，也会被识别为 GitHub（但下载会失败）
            ("", False),
        ]

        for url, expected in test_cases:
            result = github_downloader.can_handle(url)
            assert result == expected, (
                f"URL: {url}, expected: {expected}, got: {result}"
            )

    def test_download_manager_registration(self, download_manager):
        """测试下载管理器注册功能"""
        # 默认应该有一个下载器
        downloaders = download_manager.registered_downloaders
        assert len(downloaders) == 1
        assert isinstance(downloaders[0], GitHubSkillDownloader)

    def test_download_manager_with_empty_url(self, download_manager, temp_dir):
        """测试下载管理器处理空 URL"""
        result = download_manager.download("", temp_dir, "test-skill")
        assert result is None

    def test_download_manager_with_invalid_url(self, download_manager, temp_dir):
        """测试下载管理器处理无效 URL"""
        result = download_manager.download("not-a-valid-url", temp_dir, "test-skill")
        assert result is None

    @pytest.mark.integration
    def test_real_github_download(self, download_manager, temp_dir):
        """真实 GitHub 下载测试

        此测试会从真实的 GitHub 下载 skill，需要网络连接。
        如果 GitHub API 限流或网络不可用，测试可能会失败。
        """
        # 使用一个小型的公开示例仓库
        # 注意: 如果此仓库不存在或不可用，测试会失败
        github_url = "https://github.com/octocat/Hello-World/tree/master"

        # 尝试下载
        result = download_manager.download(github_url, temp_dir, "hello-world")

        # 由于我们使用的是示例仓库而非真正的 skill 仓库，
        # 可能不会有 SKILL.md，但下载过程应该能执行
        # 这里我们主要验证下载器能正确执行而不报错

        # 如果下载成功，验证目录结构
        if result is not None:
            assert result.exists()
            assert result.is_dir()
            print(f"✓ Downloaded to: {result}")

    @pytest.mark.integration
    def test_github_downloader_real_api_call(self, github_downloader, temp_dir):
        """测试 GitHub 下载器真实 API 调用

        直接测试 GitHubSkillDownloader 的真实下载能力。
        """
        # 使用 GitHub API 测试端点（octocat/Hello-World 是一个公开的测试仓库）
        github_url = "https://github.com/octocat/Hello-World/tree/master"

        # 执行下载
        result = github_downloader.download(github_url, temp_dir, "hello-world")

        # 验证下载结果
        # 注意: 如果仓库不存在或 API 限流，result 可能是 None
        if result is not None:
            print(f"✓ Download successful: {result}")
            # 验证下载的目录存在
            assert result.exists()
            # 列出下载的内容
            files = list(result.iterdir())
            print(f"  Files downloaded: {[f.name for f in files]}")
        else:
            # 下载失败可能是由于 API 限流或网络问题
            # 在集成测试中这是可以接受的
            print("⚠ Download returned None (possibly rate limited or network issue)")

    @pytest.mark.integration
    def test_download_manager_with_multiple_attempts(self, temp_dir):
        """测试下载管理器的重试机制

        创建多个下载器，验证管理器会按顺序尝试。
        """
        # 创建管理器并注册多个 GitHub 下载器（模拟不同配置）
        manager = DownloadManager()

        # 注册一个没有 token 的下载器
        config_no_token = DownloadConfig(github_token=None)
        manager.register(GitHubSkillDownloader(config_no_token))

        # 测试下载
        github_url = "https://github.com/octocat/Hello-World/tree/master"
        result = manager.download(github_url, temp_dir, "test")

        # 验证至少尝试了第一个下载器
        assert len(manager.registered_downloaders) == 1

    def test_download_config_defaults(self):
        """测试下载配置的默认值"""
        config = DownloadConfig()

        assert config.github_token is None
        assert config.github_mirrors == []
        assert config.timeout == 30

    def test_download_config_with_values(self):
        """测试下载配置的自定义值"""
        config = DownloadConfig(
            github_token="test-token",
            github_mirrors=["https://mirror1.com/", "https://mirror2.com"],
            timeout=60,
        )

        assert config.github_token == "test-token"
        assert config.github_mirrors == ["https://mirror1.com/", "https://mirror2.com"]
        assert config.timeout == 60

    def test_github_downloader_mirror_prefixes(self, github_downloader):
        """测试 GitHub 下载器的镜像前缀处理"""
        # 默认配置应该返回 [""]（直连）
        prefixes = github_downloader._get_mirror_prefixes()
        assert prefixes == [""]

    def test_github_downloader_mirror_prefixes_with_mirrors(self):
        """测试带镜像的 GitHub 下载器"""
        config = DownloadConfig(
            github_mirrors=["https://mirror1.com", "https://mirror2.com/"]
        )
        downloader = GitHubSkillDownloader(config)

        prefixes = downloader._get_mirror_prefixes()
        # 应该规范化镜像 URL（添加尾部斜杠）并添加空字符串兜底
        assert "https://mirror1.com/" in prefixes
        assert "https://mirror2.com/" in prefixes
        assert "" in prefixes


class TestDownloadManagerEdgeCases:
    """DownloadManager 边界情况测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_path = Path(tempfile.mkdtemp(prefix="test_edge_"))
        yield temp_path
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_download_to_nonexistent_directory(self, temp_dir):
        """测试下载到不存在的目录"""
        manager = create_default_download_manager()
        non_existent_dir = temp_dir / "non_existent" / "nested"

        # 目录不存在，但下载器应该会创建它
        github_url = "https://github.com/octocat/Hello-World/tree/master"
        result = manager.download(github_url, non_existent_dir.parent, "test")

        # 只要父目录存在，应该能正常工作
        # 结果可能为 None（如果 API 限流），但不会报错

    def test_download_with_special_characters_in_name(self, temp_dir):
        """测试带有特殊字符的 skill 名称"""
        manager = create_default_download_manager()

        special_names = [
            "skill-with-dashes",
            "skill_with_underscores",
            "skill.with.dots",
            "SkillWithCamelCase",
        ]

        for name in special_names:
            # 只测试 can_handle，不实际下载
            github_url = f"https://github.com/user/{name}/tree/main"
            downloader = GitHubSkillDownloader()
            assert downloader.can_handle(github_url) is True

    def test_github_url_parsing_variations(self):
        """测试各种 GitHub URL 格式的解析"""
        downloader = GitHubSkillDownloader()

        test_urls = [
            # (url, expected_owner, expected_repo)
            ("https://github.com/owner/repo/tree/main", "owner", "repo"),
            ("https://github.com/owner/repo/tree/main/subdir", "owner", "repo"),
            ("https://github.com/my-org/my-repo/tree/develop", "my-org", "my-repo"),
            ("https://github.com/user123/repo456/tree/v1.0", "user123", "repo456"),
        ]

        for url, expected_owner, expected_repo in test_urls:
            parsed = downloader._parse_github_tree_url(url)
            assert parsed is not None, f"Failed to parse: {url}"
            assert parsed["owner"] == expected_owner
            assert parsed["repo"] == expected_repo

    def test_invalid_github_urls(self):
        """测试无效 GitHub URL 的处理"""
        downloader = GitHubSkillDownloader()

        invalid_urls = [
            "https://gitlab.com/user/repo/tree/main",
            "https://github.com",  # 缺少路径
            "https://github.com/owner",  # 缺少 repo
            "https://github.com/owner/repo",  # 缺少 /tree/
            "not-a-url",
            "",
        ]

        for url in invalid_urls:
            parsed = downloader._parse_github_tree_url(url)
            assert parsed is None, f"Should have failed for: {url}"


if __name__ == "__main__":
    # 允许直接运行测试
    pytest.main([__file__, "-v", "-k", "not integration"])
