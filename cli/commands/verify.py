"""Verify command for batch skill validation."""

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

# Handle project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

console = Console()


def verify_command(
    audit_only: bool = typer.Option(False, "--audit-only", help="下载 + 仅静态审查"),
    exec_only: bool = typer.Option(False, "--exec-only", help="下载 + 仅执行验证"),
    download_only: bool = typer.Option(False, "--download-only", help="仅下载 skill"),
    sandbox: str = typer.Option("e2b", "--sandbox", help="沙箱类型: e2b / local"),
    concurrency: int = typer.Option(3, "--concurrency", "-c", help="E2B 并发数"),
    timeout: int = typer.Option(120, "--timeout", "-t", help="单个 skill 超时(秒)"),
    output: str = typer.Option(None, "--output", "-o", help="报告 JSON 输出路径"),
    test_set: str = typer.Option("test_set.jsonl", "--test-set", help="测试集路径"),
    cache_dir: str = typer.Option(
        ".verify_cache/skills", "--cache-dir", help="下载缓存目录"
    ),
    limit: int = typer.Option(None, "--limit", "-n", help="只处理前 N 个 (调试用)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
) -> None:
    """批量验证 skill: 从 test_set.jsonl 下载 + 安全审查 + 合规审查 + E2B 沙箱执行"""

    cmd = [sys.executable, str(_PROJECT_ROOT / "scripts" / "verify_pipeline.py")]

    if audit_only:
        cmd.append("--audit-only")
    elif exec_only:
        cmd.append("--exec-only")
    elif download_only:
        cmd.append("--download-only")
    else:
        cmd.append("--all")

    cmd.extend(["--sandbox", sandbox])
    cmd.extend(["--concurrency", str(concurrency)])
    cmd.extend(["--timeout", str(timeout)])
    cmd.extend(["--test-set", test_set])
    cmd.extend(["--cache-dir", cache_dir])

    if output:
        cmd.extend(["--output", output])
    if limit:
        cmd.extend(["--limit", str(limit)])
    if verbose:
        cmd.append("--verbose")

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]\n")
    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)
