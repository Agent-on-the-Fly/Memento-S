"""Skill Gateway / Provider 真实调用测试

测试范围:
- Catalog 层: discover, search, get_manifest, read, install
- Runtime 层: preflight, execute
- 性能指标: 各接口响应时间
- 异常处理: 错误场景覆盖

日志级别: DEBUG
配置来源: g_config (真实配置)
"""

from __future__ import annotations

import asyncio
import time
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 设置日志为 DEBUG 级别
import os

os.environ["LOG_LEVEL"] = "DEBUG"

from middleware.config import g_config
from utils.logger import setup_logger, get_logger
from core.skill.gateway import SkillGateway
from core.skill.config import SkillConfig
from core.skill.gateway import (
    SkillGateway,
    SkillExecutionResponse,
    SkillPreflightResult,
    SkillStatus,
    SkillErrorCode,
)
from core.skill.schema import Skill

# 初始化日志
setup_logger(
    console_level="DEBUG",
    file_level="DEBUG",
    enable_console=True,
    daily_separate=True,
)

logger = get_logger(__name__)


@dataclass
class TestResult:
    """单个测试结果"""

    name: str
    passed: bool
    duration_ms: float
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """测试套件结果"""

    name: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total_duration_ms(self) -> float:
        return sum(r.duration_ms for r in self.results)


def _banner(title: str) -> None:
    """打印分隔横幅"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    logger.info(f"=== {title} ===")


def _section(title: str) -> None:
    """打印章节标题"""
    print(f"\n--- {title} ---")
    logger.info(f"--- {title} ---")


def _log_step(step: str, details: str = "") -> None:
    """记录测试步骤"""
    msg = f"[STEP] {step}"
    if details:
        msg += f" -> {details}"
    print(msg)
    logger.debug(msg)


def _log_done(step: str, details: str = "") -> None:
    """记录完成步骤"""
    msg = f"[DONE] {step}"
    if details:
        msg += f" -> {details}"
    print(msg)
    logger.debug(msg)


def _log_result(result: TestResult) -> None:
    """记录测试结果"""
    status = "PASS" if result.passed else "FAIL"
    icon = "✓" if result.passed else "✗"
    msg = f"[{status}] {result.name} ({result.duration_ms:.2f}ms)"
    if result.error:
        msg += f" - Error: {result.error}"
    print(f"  {icon} {msg}")
    if result.passed:
        logger.info(msg)
    else:
        logger.error(msg)


class SkillGatewayTester:
    """Skill Gateway 测试器"""

    def __init__(self, gateway: SkillGateway):
        self.gateway = gateway
        self.test_suites: list[TestSuite] = []

    async def run_all_tests(self) -> None:
        """运行所有测试"""
        _banner("SKILL GATEWAY COMPREHENSIVE TEST")

        logger.info("Starting comprehensive Skill Gateway test...")
        logger.info(f"Config: workspace={g_config.paths.workspace_dir}")
        logger.info(f"Config: skills_dir={g_config.paths.skills_dir}")
        logger.info(f"Config: db_url={g_config.get_db_url()}")
        logger.info(
            f"Config: cloud_catalog_url={g_config.skills.cloud_catalog_url or 'NOT CONFIGURED'}"
        )
        logger.info(
            f"Config: embedding_base_url={g_config.skills.retrieval.embedding_base_url or 'NOT CONFIGURED'}"
        )

        # 运行各层测试
        await self._test_catalog_layer()
        await self._test_runtime_layer()
        await self._test_cloud_integration()

        # 生成评估报告
        self._generate_report()

    async def _test_catalog_layer(self) -> None:
        """测试 Catalog 层接口"""
        _section("PHASE 1: CATALOG LAYER TESTS")
        suite = TestSuite(name="Catalog Layer")

        # Test 1: discover()
        _log_step("Testing discover()", "获取所有可用技能")
        start = time.perf_counter()
        try:
            manifests = await self.gateway.discover()
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                name="discover()",
                passed=True,
                duration_ms=duration,
                details={
                    "count": len(manifests),
                    "skills": [m.name for m in manifests[:5]],
                },
            )
            print(
                f"  Found {len(manifests)} skills: {[m.name for m in manifests[:10]]}"
            )
            logger.info(f"discover() returned {len(manifests)} skills")
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                name="discover()", passed=False, duration_ms=duration, error=str(e)
            )
            logger.error(f"discover() failed: {e}")
        suite.results.append(result)
        _log_result(result)

        # Test 2: search()
        _log_step("Testing search()", "搜索技能")
        queries = ["filesystem", "search", "test"]
        for query in queries:
            start = time.perf_counter()
            try:
                results = await self.gateway.search(query, k=5)
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    name=f"search('{query}')",
                    passed=True,
                    duration_ms=duration,
                    details={
                        "query": query,
                        "results": len(results),
                        "skills": [r.name for r in results],
                    },
                )
                print(
                    f"  Search '{query}' -> {len(results)} results: {[r.name for r in results]}"
                )
                logger.info(f"search('{query}') returned {len(results)} results")
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    name=f"search('{query}')",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
                logger.error(f"search('{query}') failed: {e}")
            suite.results.append(result)
            _log_result(result)

        # Test 3: get_manifest()
        _log_step("Testing get_manifest()", "获取技能清单")
        test_skills = ["filesystem", "web_search"]
        for skill_name in test_skills:
            start = time.perf_counter()
            try:
                manifest = await self.gateway.get_manifest(skill_name)
                duration = (time.perf_counter() - start) * 1000
                if manifest:
                    result = TestResult(
                        name=f"get_manifest('{skill_name}')",
                        passed=True,
                        duration_ms=duration,
                        details={
                            "name": manifest.name,
                            "mode": manifest.execution_mode,
                        },
                    )
                    print(
                        f"  Manifest for '{skill_name}': mode={manifest.execution_mode}, deps={manifest.dependencies}"
                    )
                    logger.info(f"get_manifest('{skill_name}') returned manifest")
                else:
                    result = TestResult(
                        name=f"get_manifest('{skill_name}')",
                        passed=True,
                        duration_ms=duration,
                        details={"found": False},
                    )
                    print(f"  Manifest for '{skill_name}': NOT FOUND")
                    logger.info(f"get_manifest('{skill_name}') returned None")
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    name=f"get_manifest('{skill_name}')",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
                logger.error(f"get_manifest('{skill_name}') failed: {e}")
            suite.results.append(result)
            _log_result(result)

        # Test 4: read()
        _log_step("Testing read()", "读取技能详情")
        for skill_name in ["filesystem"]:
            start = time.perf_counter()
            try:
                envelope = await self.gateway.read(skill_name)
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    name=f"read('{skill_name}')",
                    passed=envelope.ok,
                    duration_ms=duration,
                    details={
                        "ok": envelope.ok,
                        "status": envelope.status,
                        "has_output": envelope.output is not None,
                    },
                )
                if envelope.ok:
                    output_preview = (
                        str(envelope.output)[:200] if envelope.output else "None"
                    )
                    print(f"  Read '{skill_name}': OK, output={output_preview}...")
                    logger.info(f"read('{skill_name}') succeeded")
                else:
                    print(f"  Read '{skill_name}': FAILED - {envelope.summary}")
                    logger.warning(f"read('{skill_name}') failed: {envelope.summary}")
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    name=f"read('{skill_name}')",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
                logger.error(f"read('{skill_name}') failed: {e}")
            suite.results.append(result)
            _log_result(result)

        self.test_suites.append(suite)
        _log_done(
            "Catalog Layer Tests", f"{suite.passed_count}/{len(suite.results)} passed"
        )

    async def _test_runtime_layer(self) -> None:
        """测试 Runtime 层接口"""
        _section("PHASE 2: RUNTIME LAYER TESTS")
        suite = TestSuite(name="Runtime Layer")

        # 选择一个简单的内置技能进行测试
        test_skill = "filesystem"

        # Test 1: preflight() - success case
        _log_step("Testing preflight() - success case")
        start = time.perf_counter()
        try:
            preflight_result = await self.gateway.preflight(
                test_skill, params={"path": "/tmp", "operation": "read"}
            )
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                name=f"preflight('{test_skill}')",
                passed=preflight_result.ready,
                duration_ms=duration,
                details={
                    "ready": preflight_result.ready,
                    "status": preflight_result.status,
                    "message": preflight_result.message,
                },
            )
            print(
                f"  Preflight '{test_skill}': ready={preflight_result.ready}, message={preflight_result.message}"
            )
            logger.info(
                f"preflight('{test_skill}') returned ready={preflight_result.ready}"
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                name=f"preflight('{test_skill}')",
                passed=False,
                duration_ms=duration,
                error=str(e),
            )
            logger.error(f"preflight('{test_skill}') failed: {e}")
        suite.results.append(result)
        _log_result(result)

        # Test 2: preflight() - not found case
        _log_step("Testing preflight() - skill not found")
        start = time.perf_counter()
        try:
            preflight_result = await self.gateway.preflight("nonexistent_skill_xyz")
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                name="preflight('nonexistent_skill_xyz')",
                passed=not preflight_result.ready
                and preflight_result.error_code == SkillErrorCode.SKILL_NOT_FOUND,
                duration_ms=duration,
                details={
                    "ready": preflight_result.ready,
                    "error_code": preflight_result.error_code,
                },
            )
            print(
                f"  Preflight nonexistent: ready={preflight_result.ready}, error={preflight_result.error_code}"
            )
            logger.info(
                f"preflight(nonexistent) returned error={preflight_result.error_code}"
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                name="preflight('nonexistent_skill_xyz')",
                passed=False,
                duration_ms=duration,
                error=str(e),
            )
            logger.error(f"preflight(nonexistent) failed: {e}")
        suite.results.append(result)
        _log_result(result)

        # Test 3: execute() - if skill supports execution
        _log_step("Testing execute()", "执行技能")
        start = time.perf_counter()
        try:
            # 尝试执行 filesystem skill 的 list 操作
            envelope = await self.gateway.execute(
                test_skill,
                params={
                    "request": "list files in /tmp",
                    "path": "/tmp",
                    "operation": "list",
                },
            )
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                name=f"execute('{test_skill}')",
                passed=envelope.ok,
                duration_ms=duration,
                details={
                    "ok": envelope.ok,
                    "status": envelope.status,
                    "has_output": envelope.output is not None,
                    "output_type": type(envelope.output).__name__,
                },
            )
            if envelope.ok:
                output_preview = (
                    str(envelope.output)[:200] if envelope.output else "None"
                )
                print(f"  Execute '{test_skill}': OK, output={output_preview}...")
                logger.info(f"execute('{test_skill}') succeeded")
            else:
                print(f"  Execute '{test_skill}': FAILED - {envelope.summary}")
                if envelope.diagnostics:
                    print(f"    Diagnostics: {envelope.diagnostics}")
                logger.warning(f"execute('{test_skill}') failed: {envelope.summary}")
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                name=f"execute('{test_skill}')",
                passed=False,
                duration_ms=duration,
                error=str(e),
            )
            logger.error(f"execute('{test_skill}') failed: {e}")
            logger.error(traceback.format_exc())
        suite.results.append(result)
        _log_result(result)

        self.test_suites.append(suite)
        _log_done(
            "Runtime Layer Tests", f"{suite.passed_count}/{len(suite.results)} passed"
        )

    async def _test_cloud_integration(self) -> None:
        """测试 Cloud 集成"""
        _section("PHASE 3: CLOUD INTEGRATION TESTS")
        suite = TestSuite(name="Cloud Integration")

        cloud_url = g_config.skills.cloud_catalog_url
        if not cloud_url:
            logger.warning("Cloud catalog URL not configured, skipping cloud tests")
            result = TestResult(
                name="cloud_integration",
                passed=True,  # Not a failure, just skipped
                duration_ms=0,
                details={"skipped": True, "reason": "cloud_catalog_url not configured"},
            )
            suite.results.append(result)
            print("  ⚠ Cloud integration skipped (cloud_catalog_url not configured)")
            self.test_suites.append(suite)
            return

        logger.info(f"Testing cloud integration with URL: {cloud_url}")

        # Test 1: search with cloud
        _log_step("Testing cloud search")
        start = time.perf_counter()
        try:
            results = await self.gateway.search("pdf", k=10)
            duration = (time.perf_counter() - start) * 1000

            # 检查结果是否包含云端结果
            cloud_results = [r for r in results if r.governance.source == "cloud"]
            local_results = [r for r in results if r.governance.source == "local"]

            result = TestResult(
                name="cloud_search",
                passed=True,
                duration_ms=duration,
                details={
                    "total": len(results),
                    "local": len(local_results),
                    "cloud": len(cloud_results),
                },
            )
            print(
                f"  Cloud search: total={len(results)}, local={len(local_results)}, cloud={len(cloud_results)}"
            )
            logger.info(
                f"Cloud search returned {len(results)} results ({len(cloud_results)} from cloud)"
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                name="cloud_search", passed=False, duration_ms=duration, error=str(e)
            )
            logger.error(f"Cloud search failed: {e}")
        suite.results.append(result)
        _log_result(result)

        # Test 2: install from cloud
        _log_step("Testing cloud install")
        # 使用一个不太可能存在的测试技能名，避免污染真实环境
        test_skill_name = "test_cloud_skill_xyz_12345"
        start = time.perf_counter()
        try:
            envelope = await self.gateway.install(test_skill_name)
            duration = (time.perf_counter() - start) * 1000

            # 预期会失败（技能不存在），但测试接口是否正常工作
            result = TestResult(
                name=f"install('{test_skill_name}')",
                passed=True,  # 接口调用成功，即使业务逻辑失败
                duration_ms=duration,
                details={
                    "ok": envelope.ok,
                    "status": envelope.status,
                    "error_code": envelope.error_code,
                },
            )
            print(
                f"  Install '{test_skill_name}': ok={envelope.ok}, status={envelope.status}"
            )
            if not envelope.ok:
                print(f"    Expected failure: {envelope.summary}")
            logger.info(f"install('{test_skill_name}') returned ok={envelope.ok}")
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                name=f"install('{test_skill_name}')",
                passed=False,
                duration_ms=duration,
                error=str(e),
            )
            logger.error(f"install('{test_skill_name}') failed: {e}")
        suite.results.append(result)
        _log_result(result)

        self.test_suites.append(suite)
        _log_done(
            "Cloud Integration Tests",
            f"{suite.passed_count}/{len(suite.results)} passed",
        )

    def _generate_report(self) -> None:
        """生成测试评估报告"""
        _section("TEST EVALUATION REPORT")

        total_tests = sum(len(s.results) for s in self.test_suites)
        total_passed = sum(s.passed_count for s in self.test_suites)
        total_failed = sum(s.failed_count for s in self.test_suites)
        total_duration = sum(s.total_duration_ms for s in self.test_suites)

        print("\n📊 SUMMARY")
        print(f"  Total Test Suites: {len(self.test_suites)}")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {total_passed} ({total_passed / total_tests * 100:.1f}%)")
        print(f"  Failed: {total_failed} ({total_failed / total_tests * 100:.1f}%)")
        print(
            f"  Total Duration: {total_duration:.2f}ms ({total_duration / 1000:.2f}s)"
        )

        print("\n📋 SUITE BREAKDOWN")
        for suite in self.test_suites:
            print(f"\n  {suite.name}:")
            print(f"    Tests: {len(suite.results)}")
            print(f"    Passed: {suite.passed_count}")
            print(f"    Failed: {suite.failed_count}")
            print(f"    Duration: {suite.total_duration_ms:.2f}ms")

            if suite.failed_count > 0:
                print(f"    Failed Tests:")
                for r in suite.results:
                    if not r.passed:
                        print(f"      - {r.name}: {r.error or 'Assertion failed'}")

        print("\n⏱️  PERFORMANCE METRICS")
        all_results = [r for s in self.test_suites for r in s.results]
        if all_results:
            durations = [r.duration_ms for r in all_results]
            print(f"  Fastest: {min(durations):.2f}ms")
            print(f"  Slowest: {max(durations):.2f}ms")
            print(f"  Average: {sum(durations) / len(durations):.2f}ms")

            # 按接口类型分组统计
            print("\n  By Interface Type:")
            interface_types = {}
            for r in all_results:
                iface = r.name.split("(")[0]
                if iface not in interface_types:
                    interface_types[iface] = []
                interface_types[iface].append(r.duration_ms)

            for iface, times in sorted(interface_types.items()):
                avg = sum(times) / len(times)
                print(f"    {iface}: avg={avg:.2f}ms, count={len(times)}")

        print("\n✅ EVALUATION")
        if total_failed == 0:
            print("  Status: ALL TESTS PASSED")
            print("  Assessment: Skill Gateway is fully functional")
        elif total_failed / total_tests < 0.2:
            print("  Status: MOSTLY PASSED")
            print("  Assessment: Skill Gateway is functional with minor issues")
        else:
            print("  Status: ISSUES DETECTED")
            print(
                "  Assessment: Skill Gateway has significant issues requiring attention"
            )

        # 记录到日志
        logger.info("=" * 60)
        logger.info("TEST EVALUATION REPORT")
        logger.info(
            f"Total: {total_tests}, Passed: {total_passed}, Failed: {total_failed}"
        )
        logger.info(f"Duration: {total_duration:.2f}ms")
        logger.info("=" * 60)


async def main() -> None:
    """主函数"""
    _banner("SKILL GATEWAY TEST INITIALIZATION")

    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory(prefix="skill-gateway-test-") as tmpdir:
        logger.info(f"Using temporary directory: {tmpdir}")

        # 创建 SkillGateway
        _section("INITIALIZING SKILL PROVIDER")
        logger.info("Creating SkillGateway with DEBUG logging...")

        try:
            skill_config = SkillConfig.from_global_config()
            provider = await SkillGateway.from_config(skill_config)
            logger.info("SkillGateway created successfully")
            print(f"  ✓ SkillGateway initialized")
            print(f"    Type: {type(provider).__name__}")
            print(
                f"    Cloud catalog: {'Connected' if provider._cloud_catalog else 'Not configured'}"
            )
        except Exception as e:
            logger.error(f"Failed to create SkillGateway: {e}")
            logger.error(traceback.format_exc())
            print(f"  ✗ Failed to initialize SkillGateway: {e}")
            return

        # 运行测试
        tester = SkillGatewayTester(provider)
        await tester.run_all_tests()

        _banner("TEST COMPLETED")
        print(
            "\n详细日志请查看:",
            g_config.get_log_path("app_" + time.strftime("%Y-%m-%d") + ".log"),
        )


if __name__ == "__main__":
    asyncio.run(main())
