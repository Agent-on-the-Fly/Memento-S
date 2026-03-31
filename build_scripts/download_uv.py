#!/usr/bin/env python3
"""
打包前下载 uv 二进制 + Python standalone 到 resources/

用法：
  python build_scripts/download_uv.py             # 仅下载 uv
  python build_scripts/download_uv.py --with-python  # 同时下载 Python standalone
  python build_scripts/download_uv.py --uv-version 0.5.1 --python-version 3.12.9

下载后的文件位置：
  resources/bin/uv[.exe]          ← uv 可执行文件
  resources/python/python[.exe]   ← Python 解释器（--with-python 时）
"""

import argparse
import platform
import shutil
import stat
import sys
import urllib.request
from pathlib import Path

# ──────────────────────────────────────────────
# uv
# ──────────────────────────────────────────────
_UV_DOWNLOAD_URL = (
    "https://github.com/astral-sh/uv/releases/download/{version}/{filename}"
)

_UV_PLATFORM_FILENAMES: dict[tuple[str, str], str] = {
    ("Windows", "AMD64"):  "uv-x86_64-pc-windows-msvc.zip",
    ("Windows", "ARM64"):  "uv-aarch64-pc-windows-msvc.zip",
    ("Darwin",  "x86_64"): "uv-x86_64-apple-darwin.tar.gz",
    ("Darwin",  "arm64"):  "uv-aarch64-apple-darwin.tar.gz",
    ("Linux",   "x86_64"): "uv-x86_64-unknown-linux-gnu.tar.gz",
    ("Linux",   "aarch64"): "uv-aarch64-unknown-linux-gnu.tar.gz",
}

# ──────────────────────────────────────────────
# Python standalone (indygreg/python-build-standalone)
# ──────────────────────────────────────────────
_PYTHON_DOWNLOAD_URL = (
    "https://github.com/indygreg/python-build-standalone/releases/download"
    "/{build_date}/cpython-{version}+{build_date}-{triple}-install_only_stripped.tar.gz"
)

# 各平台对应的 triple 和默认构建日期
_PYTHON_PLATFORM_TRIPLES: dict[tuple[str, str], str] = {
    ("Windows", "AMD64"):  "x86_64-pc-windows-msvc",
    ("Windows", "ARM64"):  "aarch64-pc-windows-msvc",
    ("Darwin",  "x86_64"): "x86_64-apple-darwin",
    ("Darwin",  "arm64"):  "aarch64-apple-darwin",
    ("Linux",   "x86_64"): "x86_64-unknown-linux-gnu",
    ("Linux",   "aarch64"): "aarch64-unknown-linux-gnu",
}

# Python standalone 发布构建日期，与版本号配套（可按需更新）
_PYTHON_DEFAULT_BUILD_DATE = "20260303"
_PYTHON_DEFAULT_VERSION = "3.12.9"


# ──────────────────────────────────────────────
# 通用工具
# ──────────────────────────────────────────────

def _platform_key() -> tuple[str, str]:
    os_name = platform.system()
    machine = platform.machine()
    # Windows AMD64 在某些系统报告为 x86_64
    if os_name == "Windows" and machine == "x86_64":
        machine = "AMD64"
    return os_name, machine


def _progress_hook(label: str):
    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            print(f"\r[INFO] {label} {pct}% ({mb:.1f} MB)", end="", flush=True)
    return _hook


def _get_latest_uv_version() -> str:
    """从 GitHub API 获取最新 uv 版本号。"""
    import json
    api_url = "https://api.github.com/repos/astral-sh/uv/releases/latest"
    req = urllib.request.Request(api_url, headers={"User-Agent": "download_uv.py"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data["tag_name"].lstrip("v")


# ──────────────────────────────────────────────
# 下载 uv
# ──────────────────────────────────────────────

def download_uv(version: str, dest_dir: Path) -> Path:
    """下载并解压 uv 到 dest_dir/uv[.exe]，返回可执行文件路径。"""
    os_name, machine = _platform_key()
    filename = _UV_PLATFORM_FILENAMES.get((os_name, machine))
    if not filename:
        raise RuntimeError(f"uv: 不支持的平台 {os_name}/{machine}")

    url = _UV_DOWNLOAD_URL.format(version=version, filename=filename)
    archive_path = dest_dir / filename
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 下载 uv v{version}  ({os_name}/{machine})")
    print(f"[INFO] {url}")
    urllib.request.urlretrieve(url, archive_path, reporthook=_progress_hook("uv 下载中..."))
    print()

    uv_name = "uv.exe" if os_name == "Windows" else "uv"
    uv_path = dest_dir / uv_name

    if filename.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                if member.endswith(uv_name) and not member.endswith("uvx.exe"):
                    uv_path.write_bytes(zf.read(member))
                    break
            else:
                raise RuntimeError(f"uv 归档中未找到 {uv_name}")
    else:
        import tarfile
        with tarfile.open(archive_path) as tf:
            for member in tf.getmembers():
                if member.name.endswith("/" + uv_name) or member.name == uv_name:
                    f = tf.extractfile(member)
                    if f:
                        uv_path.write_bytes(f.read())
                        break
            else:
                raise RuntimeError(f"uv 归档中未找到 {uv_name}")

    archive_path.unlink()
    if os_name != "Windows":
        uv_path.chmod(uv_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return uv_path


# ──────────────────────────────────────────────
# 下载 Python standalone
# ──────────────────────────────────────────────

def download_python(
    version: str,
    dest_dir: Path,
    build_date: str = _PYTHON_DEFAULT_BUILD_DATE,
) -> Path:
    """下载 Python standalone 并解压到 dest_dir/，返回 python 可执行文件路径。

    解压后目录结构（install_only_stripped）：
      Windows : dest_dir/python.exe
      Unix    : dest_dir/bin/python3
    """
    os_name, machine = _platform_key()
    triple = _PYTHON_PLATFORM_TRIPLES.get((os_name, machine))
    if not triple:
        raise RuntimeError(f"Python standalone: 不支持的平台 {os_name}/{machine}")

    url = _PYTHON_DOWNLOAD_URL.format(
        version=version, build_date=build_date, triple=triple
    )
    archive_name = url.split("/")[-1]
    archive_path = dest_dir / archive_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 下载 Python {version} standalone ({os_name}/{machine})")
    print(f"[INFO] {url}")
    urllib.request.urlretrieve(url, archive_path, reporthook=_progress_hook("Python 下载中..."))
    print()

    # 解压：archive 内为 python/install/...，解压到 dest_dir
    import tarfile
    print("[INFO] 解压 Python standalone ...")
    with tarfile.open(archive_path) as tf:
        # 只提取 python/install/ 下的内容，去掉前缀路径
        members = []
        prefix = "python/install/"
        for m in tf.getmembers():
            if m.name.startswith(prefix):
                # 重写目标路径，去掉 python/install/ 前缀
                m.name = m.name[len(prefix):]
                if m.name:  # 跳过根目录本身
                    members.append(m)
        tf.extractall(dest_dir, members=members)  # noqa: S202 (trusted local archive)

    archive_path.unlink()

    # 返回 Python 可执行文件路径
    if os_name == "Windows":
        python_path = dest_dir / "python.exe"
    else:
        # Unix: bin/python3.x 和 bin/python3 软链
        python_path = dest_dir / "bin" / "python3"
        # 确保可执行权限
        if python_path.exists():
            python_path.chmod(python_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        # 同步修复 bin/ 下所有文件权限
        bin_dir = dest_dir / "bin"
        if bin_dir.exists():
            for f in bin_dir.iterdir():
                if f.is_file():
                    f.chmod(f.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    if not python_path.exists():
        raise RuntimeError(f"解压后未找到 Python 可执行文件: {python_path}")

    return python_path


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载 uv 和/或 Python standalone 到 resources/ 目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python build_scripts/download_uv.py                        # 仅下载 uv（自动获取最新版）
  python build_scripts/download_uv.py --with-python          # 同时下载 Python standalone
  python build_scripts/download_uv.py --uv-version 0.6.1 --with-python --python-version 3.12.9
        """,
    )
    parser.add_argument("--uv-version", default=None, help="uv 版本（默认：自动获取最新）")
    parser.add_argument("--with-python", action="store_true", help="同时下载 Python standalone")
    parser.add_argument(
        "--python-version", default=_PYTHON_DEFAULT_VERSION,
        help=f"Python 版本（默认：{_PYTHON_DEFAULT_VERSION}）",
    )
    parser.add_argument(
        "--python-build-date", default=_PYTHON_DEFAULT_BUILD_DATE,
        help=f"python-build-standalone 构建日期（默认：{_PYTHON_DEFAULT_BUILD_DATE}）",
    )
    parser.add_argument("--dest", default=None, help="项目根目录（默认：脚本上级目录）")
    args = parser.parse_args()

    project_root = Path(args.dest) if args.dest else Path(__file__).resolve().parent.parent
    bin_dir = project_root / "resources" / "bin"
    python_dir = project_root / "resources" / "python"

    # ── 下载 uv ──
    uv_name = "uv.exe" if platform.system() == "Windows" else "uv"
    uv_path = bin_dir / uv_name

    uv_version = args.uv_version
    if not uv_version:
        print("[INFO] 正在获取最新 uv 版本...")
        uv_version = _get_latest_uv_version()
        print(f"[INFO] 最新 uv 版本: {uv_version}")

    if uv_path.exists() and not args.uv_version:
        print(f"[INFO] uv 已存在: {uv_path}（跳过，指定 --uv-version 可强制重下）")
    else:
        uv_path = download_uv(uv_version, bin_dir)
        print(f"[SUCCESS] uv → {uv_path} ({uv_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ── 下载 Python standalone（可选）──
    if args.with_python:
        os_name = platform.system()
        python_exe = "python.exe" if os_name == "Windows" else "bin/python3"
        python_path = python_dir / python_exe

        if python_path.exists() and not args.python_version:
            print(f"[INFO] Python 已存在: {python_path}（跳过，指定 --python-version 可强制重下）")
        else:
            if python_dir.exists():
                shutil.rmtree(python_dir)  # 清理旧版本
            python_path = download_python(
                args.python_version, python_dir, args.python_build_date
            )
            total_mb = sum(
                f.stat().st_size for f in python_dir.rglob("*") if f.is_file()
            ) / 1024 / 1024
            print(f"[SUCCESS] Python → {python_path} (目录共 {total_mb:.1f} MB)")

    print("\n[INFO] 完成！现在可以运行打包脚本。")


if __name__ == "__main__":
    main()
