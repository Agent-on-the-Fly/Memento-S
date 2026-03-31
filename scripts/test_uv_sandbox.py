"""Manual end-to-end tests for UvLocalSandbox.

Run:
  python scripts/test_uv_sandbox.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from middleware.config import g_config
from middleware.sandbox import UvLocalSandbox
from core.skill.schema import Skill
from middleware.sandbox.artifacts import ArtifactManager


@dataclass
class CaseResult:
    name: str
    success: bool
    result: str | None
    error: str | None
    artifacts: list[str]
    duration_ms: int
    work_dir: str | None = None
    output_dir: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


class Recorder:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.workspace = Path(g_config.paths.workspace_dir).resolve()
        self.session_root = self.workspace / "sessions" / session_id
        self.log_dir = self.session_root / "test_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_case(self, case: CaseResult) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "timestamp": ts,
            "name": case.name,
            "success": case.success,
            "result": case.result,
            "error": case.error,
            "artifacts": case.artifacts,
            "duration_ms": case.duration_ms,
            "work_dir": case.work_dir,
            "output_dir": case.output_dir,
            "extra": case.extra,
        }
        log_path = self.log_dir / f"{case.name}.json"
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


class CaseRunner:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.sandbox = UvLocalSandbox()
        self.recorder = Recorder(session_id)

    def run_case(
        self,
        name: str,
        code: str,
        *,
        skill_name: str = "uv_sandbox_test",
        deps: list[str] | None = None,
        session_id: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        record_paths: bool = True,
    ) -> CaseResult:
        actual_session = session_id or self.session_id
        skill = Skill(
            name=skill_name,
            description="",
            instruction="",
            code=code,  # code is required field
        )
        start = time.time()
        result = self.sandbox.run_code(
            code,
            skill=skill,
            deps=deps,
            session_id=actual_session,
        )
        elapsed_ms = int((time.time() - start) * 1000)

        case_result = CaseResult(
            name=name,
            success=result.success,
            result=result.result
            if isinstance(result.result, str)
            else str(result.result),
            error=result.error,
            artifacts=result.artifacts or [],
            duration_ms=elapsed_ms,
        )

        if record_paths:
            work_dir = ArtifactManager.get_sandbox_dir(skill.name, actual_session)
            output_dir = ArtifactManager.get_output_dir(skill.name, actual_session)
            case_result.work_dir = str(work_dir)
            case_result.output_dir = str(output_dir)

        self.recorder.log_case(case_result)
        return case_result

    def run_raw_command_case(
        self,
        name: str,
        cmd: list[str],
        *,
        cwd: Path,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> CaseResult:
        start = time.time()
        result = self.sandbox.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            env=env,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        case_result = CaseResult(
            name=name,
            success=result.success,
            result=result.result
            if isinstance(result.result, str)
            else str(result.result),
            error=result.error,
            artifacts=result.artifacts or [],
            duration_ms=elapsed_ms,
            work_dir=str(cwd),
        )
        self.recorder.log_case(case_result)
        return case_result


def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_case(case: CaseResult) -> None:
    print(f"[CASE] {case.name}")
    print(f"  success: {case.success}")
    print(f"  duration_ms: {case.duration_ms}")
    if case.work_dir:
        print(f"  work_dir: {case.work_dir}")
    if case.output_dir:
        print(f"  output_dir: {case.output_dir}")
    if case.result:
        print("  result:")
        print(f"{case.result}")
    if case.error:
        print("  error:")
        print(f"{case.error}")
    if case.artifacts:
        print("  artifacts:")
        for art in case.artifacts:
            print(f"    - {art}")
    print("-")


def ensure_config_loaded() -> None:
    """确保配置已加载"""
    try:
        # 尝试访问配置，如果未加载会抛出异常
        _ = g_config.paths.workspace_dir
    except RuntimeError:
        # 配置未加载，尝试加载
        try:
            g_config.load()
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")


def build_cases(runner: CaseRunner) -> list[CaseResult]:
    cases: list[CaseResult] = []

    # A. 基础执行路径
    cases.append(
        runner.run_case(
            "A1_hello_world",
            "print('hello')",
        )
    )

    cases.append(
        runner.run_case(
            "A2_raise_error",
            "raise ValueError('boom')",
        )
    )

    cases.append(
        runner.run_case(
            "A3_syntax_error",
            "print(",
        )
    )

    # B. 产物收集
    cases.append(
        runner.run_case(
            "B1_create_file",
            "from pathlib import Path\nPath('out.txt').write_text('ok')\nprint('done')",
        )
    )

    cases.append(
        runner.run_case(
            "B2_nested_file",
            "from pathlib import Path\nPath('a/b/c.json').parent.mkdir(parents=True, exist_ok=True)\n"
            "Path('a/b/c.json').write_text('{}')\nprint('nested')",
        )
    )

    cases.append(
        runner.run_case(
            "B3_ignore_init",
            "from pathlib import Path\nPath('__init__.py').write_text('x')\nprint('init')",
        )
    )

    # C. session & workspace 路径
    cases.append(
        runner.run_case(
            "C1_session_default",
            "print('default session')",
            session_id="",
        )
    )

    cases.append(
        runner.run_case(
            "C2_session_specific",
            "print('case1 session')",
            session_id="case1",
        )
    )

    # C3: workspace 范围校验（通过 run 指定 cwd 到 workspace 外）
    workspace = Path(g_config.paths.workspace_dir).resolve()
    outside = Path(tempfile.gettempdir()).resolve() / "uv_sandbox_outside"
    outside.mkdir(parents=True, exist_ok=True)
    if outside.is_relative_to(workspace):
        outside = Path.home().resolve()
    cases.append(
        runner.run_raw_command_case(
            "C3_cwd_outside_workspace_should_fail",
            [sys.executable, "-c", "print('outside')"],
            cwd=outside,
        )
    )

    # C4: 系统临时目录允许
    cases.append(
        runner.run_raw_command_case(
            "C4_cwd_tmp_allowed",
            [sys.executable, "-c", "print('tmp ok')"],
            cwd=Path(tempfile.gettempdir()).resolve(),
        )
    )

    # D. 依赖安装
    cases.append(
        runner.run_case(
            "D1_install_requests",
            "import requests\nprint(requests.__version__)",
            deps=["requests"],
        )
    )

    cases.append(
        runner.run_case(
            "D2_install_failure",
            "print('this should not run')",
            deps=["nonexistent_pkg_12345"],
        )
    )

    # E. CLI 工具安装（可选，注意会改变本机环境）
    # 可根据需要手动开启，默认跳过

    # F. 超时控制
    cases.append(
        runner.run_raw_command_case(
            "F1_timeout",
            [sys.executable, "-c", "import time; time.sleep(999)"],
            cwd=Path(tempfile.gettempdir()).resolve(),
            timeout=1,
        )
    )

    # G. 连续执行
    for i in range(5):
        cases.append(
            runner.run_case(
                f"G1_repeat_{i + 1}",
                "print('repeat')",
            )
        )

    # Extra cases
    cases.append(
        runner.run_case(
            "X1_env_passthrough",
            "import os\nprint(os.environ.get('TEST_ENV_KEY', 'missing'))",
        )
    )

    cases.append(
        runner.run_raw_command_case(
            "X2_env_injection",
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('TEST_ENV_KEY', 'missing'))",
            ],
            cwd=Path(tempfile.gettempdir()).resolve(),
            env={"TEST_ENV_KEY": "OK_FROM_ENV"},
        )
    )

    cases.append(
        runner.run_case(
            "X3_config_envs",
            "import json\nimport os\n\nkeys = [k for k in os.environ.keys() if k.isupper()]\nprint(json.dumps({k: os.environ.get(k) for k in sorted(keys)}, ensure_ascii=False))",
        )
    )

    cases.append(
        runner.run_case(
            "X4_output_large",
            "print('x' * 5000)",
        )
    )

    cases.append(
        runner.run_case(
            "X5_create_binary_file",
            "from pathlib import Path\nPath('bin.dat').write_bytes(b'\\x00\\x01\\x02')\nprint('bin')",
        )
    )

    return cases


def main() -> None:
    ensure_config_loaded()

    session_id = "uv_sandbox_test"
    runner = CaseRunner(session_id=session_id)

    print_header("UvLocalSandbox end-to-end test")
    print(f"workspace_dir: {g_config.paths.workspace_dir}")
    print(f"session_id: {session_id}")
    print("-")

    cases = build_cases(runner)

    print_header("Results")
    success_count = 0
    for case in cases:
        print_case(case)
        if case.success:
            success_count += 1

    total = len(cases)
    print_header("Summary")
    print(f"Total: {total}")
    print(f"Success: {success_count}")
    print(f"Failed: {total - success_count}")
    print(f"Log dir: {runner.recorder.log_dir}")


if __name__ == "__main__":
    main()
