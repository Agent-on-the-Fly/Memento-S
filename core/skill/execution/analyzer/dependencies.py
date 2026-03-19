"""依赖分析 — 依赖检查 / 依赖解析（公共工具函数）"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import shutil

from utils.logger import get_logger

logger = get_logger(__name__)

_VERSION_EXTRAS_RE = re.compile(r"[\[=<>!~].*$")

_MISSING_MODULE_PATTERNS = [
    re.compile(r"no module named ['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"module not found: ['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"no module named ([^\s]+)", re.IGNORECASE),
    re.compile(r"importerror: cannot import name ['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"cannot import name ['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"module\s+['\"]([^'\"]+)['\"]\s+has no attribute", re.IGNORECASE),
]


# ---- public helpers ----


def strip_version_extras(spec: str) -> str:
    """Strip version specifiers and extras from a requirement string.

    "httpx[http2]>=0.24" -> "httpx"
    "scikit-learn>=1.0"  -> "scikit-learn"
    """
    return _VERSION_EXTRAS_RE.sub("", spec).strip()


def parse_dependency(dep: str) -> tuple[str, str, str]:
    """Parse dependency spec into (kind, name, install_spec).

    Returns:
        kind: "python" | "cli" | "none"
        name: package/module name for checking (e.g. "sklearn", "scikit-learn")
        install_spec: raw spec for pip install (e.g. "scikit-learn>=1.0", "httpx[http2]")

    Supported forms:
    - cli:<tool>    -> CLI tool via PATH check
    - pip:<package> -> Python package install/check
    - py:<module>   -> Python module check, install by same name
    - plain spec    -> infer from requirement string
    """
    raw = (dep or "").strip()
    if not raw:
        return ("none", "", "")

    lowered = raw.lower()
    if lowered.startswith("cli:"):
        tool = raw.split(":", 1)[1].strip()
        return ("cli", tool, tool)
    if lowered.startswith("pip:"):
        pkg = raw.split(":", 1)[1].strip()
        base = strip_version_extras(pkg)
        return ("python", base, pkg)
    if lowered.startswith("py:"):
        mod = raw.split(":", 1)[1].strip()
        return ("python", mod, mod)

    base = strip_version_extras(raw)
    if not base:
        return ("none", "", "")

    if base.lower() in {"ffmpeg"}:
        return ("cli", base, base)

    return ("python", base, raw)


def is_installed(name: str) -> bool:
    """Check if a Python package/module is available.

    Tries find_spec first (works for import names like 'sklearn'),
    then distribution() (works for PyPI names like 'scikit-learn').
    """
    try:
        if importlib.util.find_spec(name) is not None:
            return True
    except (ModuleNotFoundError, ValueError):
        pass
    try:
        importlib.metadata.distribution(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def check_missing_dependencies(dependencies: list[str]) -> list[str]:
    """Check which dependencies are missing."""
    missing: list[str] = []
    for dep in dependencies:
        kind, name, _ = parse_dependency(dep)
        if kind == "none" or not name:
            continue
        if kind == "cli":
            if shutil.which(name) is None:
                missing.append(dep)
            continue
        if not is_installed(name):
            missing.append(dep)
    return missing


def extract_missing_module_from_error(error_text: str) -> str | None:
    """从运行时报错输出中提取缺失的模块名。

    支持 ModuleNotFoundError / ImportError 等常见格式。
    返回顶层模块名（去掉子模块），找不到则返回 None。
    """
    if not error_text:
        return None
    for pattern in _MISSING_MODULE_PATTERNS:
        match = pattern.search(error_text)
        if not match:
            continue
        name = match.group(1).strip()
        if not name:
            continue
        if "." in name:
            name = name.split(".", 1)[0]
        return name
    return None

