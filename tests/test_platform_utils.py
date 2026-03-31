#!/usr/bin/env python3
"""
测试: platform_utils 平台能力探测

验证 platform_utils 在当前平台上的探测结果是否正确、可用。
不使用 mock，直接运行并打印探测到的路径和值。

使用方法:
    .venv/bin/python tests/test_platform_utils.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "platform_utils",
    project_root / "core" / "skill" / "execution" / "platform_utils.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SCRIPT_EXTENSIONS = _mod.SCRIPT_EXTENSIONS
background_hint = _mod.background_hint
chmod_executable = _mod.chmod_executable
filter_env_by_whitelist = _mod.filter_env_by_whitelist
has_bash = _mod.has_bash
has_powershell = _mod.has_powershell
is_path_within = _mod.is_path_within
pip_shim_content = _mod.pip_shim_content
pip_shim_path = _mod.pip_shim_path
python_executable = _mod.python_executable
temp_dir = _mod.temp_dir
uv_install_hint = _mod.uv_install_hint
venv_bin_dir = _mod.venv_bin_dir
venv_python = _mod.venv_python


# ================================================================
# 1. python_executable
# ================================================================

def test_python_executable():
    print("\n【1. python_executable】")
    py = python_executable()
    print(f"  路径: {py}")
    assert Path(py).exists(), f"Python 路径不存在: {py}"
    print(f"  存在: True")

    result = subprocess.run(
        [py, "--version"], capture_output=True,
        encoding="utf-8", errors="replace",
    )
    version = result.stdout.strip() or result.stderr.strip()
    print(f"  版本: {version}")
    assert result.returncode == 0, f"执行失败: {result.stderr}"
    print("  ✓ python_executable 可用")


# ================================================================
# 2. temp_dir
# ================================================================

def test_temp_dir():
    print("\n【2. temp_dir】")
    td = temp_dir()
    print(f"  路径: {td}")
    assert Path(td).is_dir(), f"临时目录不存在: {td}"
    print(f"  存在: True")

    expected = tempfile.gettempdir()
    assert td == expected, f"不一致: {td} != {expected}"
    print(f"  与 tempfile.gettempdir() 一致: True")
    print("  ✓ temp_dir 正确")


# ================================================================
# 3. venv 布局探测
# ================================================================

def test_venv_layout():
    print("\n【3. venv 布局探测】")

    fake_venv = Path(tempfile.mkdtemp()) / "test_venv"

    bin_dir = venv_bin_dir(fake_venv)
    py_path = venv_python(fake_venv)

    print(f"  venv_bin_dir: {bin_dir.name}")
    print(f"  venv_python:  {py_path.name}")

    assert bin_dir.name in ("bin", "Scripts"), f"意外的 bin 目录名: {bin_dir.name}"
    assert py_path.name in ("python", "python.exe", "python3"), f"意外的 python 名: {py_path.name}"

    if os.name == "posix":
        assert bin_dir.name == "bin", "POSIX 上应为 bin"
        print("  ✓ POSIX 布局正确 (bin/python)")
    else:
        assert bin_dir.name == "Scripts", "Windows 上应为 Scripts"
        print("  ✓ Windows 布局正确 (Scripts/python.exe)")


# ================================================================
# 4. SCRIPT_EXTENSIONS
# ================================================================

def test_script_extensions():
    print("\n【4. SCRIPT_EXTENSIONS】")
    print(f"  探测到: {sorted(SCRIPT_EXTENSIONS)}")

    assert ".py" in SCRIPT_EXTENSIONS, ".py 必须在其中"

    pathext = os.environ.get("PATHEXT", "")
    if pathext:
        print(f"  PATHEXT 存在: {pathext}")
        assert ".sh" not in SCRIPT_EXTENSIONS, "有 PATHEXT 时不应包含 .sh"
        print("  ✓ Windows 风格后缀（从 PATHEXT 探测）")
    else:
        print("  PATHEXT 不存在")
        assert ".sh" in SCRIPT_EXTENSIONS, "无 PATHEXT 时应包含 .sh"
        print("  ✓ POSIX 风格后缀（.py, .sh）")


# ================================================================
# 5. chmod_executable
# ================================================================

def test_chmod_executable():
    print("\n【5. chmod_executable】")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sh") as f:
        f.write(b"#!/bin/sh\necho ok\n")
        tmp_path = Path(f.name)

    try:
        chmod_executable(tmp_path)
        if os.name == "posix":
            mode = oct(tmp_path.stat().st_mode)
            print(f"  文件权限: {mode}")
            assert tmp_path.stat().st_mode & 0o111, "POSIX 上应有执行权限"
            print("  ✓ POSIX chmod 生效")
        else:
            print("  ✓ Windows 上 chmod 为空操作（正常）")
    finally:
        tmp_path.unlink()


# ================================================================
# 6. Shell 能力探测
# ================================================================

def test_shell_detection():
    print("\n【6. Shell 能力探测】")

    bash = has_bash()
    ps = has_powershell()
    print(f"  has_bash:       {bash}")
    print(f"  has_powershell: {ps}")

    hint_bg = background_hint()
    hint_uv = uv_install_hint()
    print(f"  background_hint:  '{hint_bg}'")
    print(f"  uv_install_hint:  '{hint_uv}'")

    if bash:
        assert "nohup" in hint_bg
        result = subprocess.run(
            ["bash", "--version"], capture_output=True,
            encoding="utf-8", errors="replace",
        )
        print(f"  bash 版本: {result.stdout.splitlines()[0] if result.stdout else 'N/A'}")
    else:
        assert "start /b" in hint_bg

    print("  ✓ Shell 探测与提示文案一致")


# ================================================================
# 7. is_path_within
# ================================================================

def test_is_path_within():
    print("\n【7. is_path_within】")

    with tempfile.TemporaryDirectory() as td:
        parent = Path(td)
        child = parent / "sub" / "file.txt"
        outside = Path(tempfile.gettempdir()).parent / "outside"

        r1 = is_path_within(child, parent)
        r2 = is_path_within(outside, parent)

        print(f"  parent:  {parent}")
        print(f"  child:   {child}  -> within: {r1}")
        print(f"  outside: {outside} -> within: {r2}")

        assert r1 is True, "子路径应在父目录内"
        assert r2 is False, "外部路径不应在父目录内"

    print("  ✓ 路径包含检查正确")


# ================================================================
# 8. filter_env_by_whitelist
# ================================================================

def test_filter_env_by_whitelist():
    print("\n【8. filter_env_by_whitelist】")

    test_env = {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "USERPROFILE": "C:\\Users\\test",
        "SECRET_KEY": "should_not_pass",
        "MEMENTO_FOO": "bar",
        "UV_CACHE_DIR": "/tmp/uv",
        "RANDOM_VAR": "nope",
    }

    filtered = filter_env_by_whitelist(test_env)
    print(f"  输入: {list(test_env.keys())}")
    print(f"  输出: {list(filtered.keys())}")

    assert "PATH" in filtered
    assert "HOME" in filtered
    assert "USERPROFILE" in filtered
    assert "MEMENTO_FOO" in filtered, "MEMENTO_* 通配符应匹配"
    assert "UV_CACHE_DIR" in filtered, "UV_* 通配符应匹配"
    assert "SECRET_KEY" not in filtered, "非白名单变量不应通过"
    assert "RANDOM_VAR" not in filtered

    real_filtered = filter_env_by_whitelist()
    print(f"  当前系统白名单变量数: {len(real_filtered)}")
    assert "PATH" in real_filtered
    print("  ✓ 白名单过滤正确")


# ================================================================
# 9. pip shim
# ================================================================

def test_pip_shim():
    print("\n【9. pip shim】")

    fake_venv = Path(tempfile.mkdtemp()) / "test_venv"
    fake_python = Path("/usr/bin/python3")

    shim_path = pip_shim_path(fake_venv)
    shim_content = pip_shim_content(fake_python)

    print(f"  pip shim 文件名: {shim_path.name}")
    print(f"  pip shim 内容:\n    {shim_content.strip()}")

    if os.name == "posix":
        assert shim_path.name == "pip", "POSIX 应为 pip"
        assert "#!/bin/sh" in shim_content
        assert "exec" in shim_content
        print("  ✓ POSIX pip shim 正确")
    else:
        assert shim_path.name == "pip.bat", "Windows 应为 pip.bat"
        assert "@echo off" in shim_content
        print("  ✓ Windows pip shim 正确")


# ================================================================
# 10. 综合：当前平台摘要
# ================================================================

def test_platform_summary():
    print("\n【10. 当前平台摘要】")
    print(f"  os.name:            {os.name}")
    print(f"  sys.platform:       {sys.platform}")
    print(f"  python_executable:  {python_executable()}")
    print(f"  temp_dir:           {temp_dir()}")
    print(f"  venv_bin_dir name:  {venv_bin_dir(Path('/fake')).name}")
    print(f"  venv_python name:   {venv_python(Path('/fake')).name}")
    print(f"  SCRIPT_EXTENSIONS:  {sorted(SCRIPT_EXTENSIONS)}")
    print(f"  has_bash:           {has_bash()}")
    print(f"  has_powershell:     {has_powershell()}")
    print(f"  PATHEXT:            {os.environ.get('PATHEXT', '(not set)')}")
    print("  ✓ 摘要输出完成")


if __name__ == "__main__":
    print("=" * 70)
    print("测试 platform_utils — 平台能力探测")
    print("=" * 70)

    test_python_executable()
    test_temp_dir()
    test_venv_layout()
    test_script_extensions()
    test_chmod_executable()
    test_shell_detection()
    test_is_path_within()
    test_filter_env_by_whitelist()
    test_pip_shim()
    test_platform_summary()

    print("\n" + "=" * 70)
    print("✓ 所有测试通过！")
    print("=" * 70)
