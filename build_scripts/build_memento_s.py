#!/usr/bin/env python3
"""
flet pack 打包脚本
直接调用 flet_cli.commands.pack.Command，绕过顶层 flet CLI 解析器的限制
（flet CLI 使用 parse_args() 而非 parse_known_args()，导致传给
 --pyinstaller-build-args 的 -- 开头参数被误判为未知参数）
"""

import argparse
import sys
import platform
import importlib.util
from pathlib import Path


class MementoSPackBuilder:
    """memento-s pack 构建器"""

    _BASE_APP_NAME = "memento-s"
    PRODUCT_NAME = "Memento-S"

    # 平台缩写映射
    _PLATFORM_MAP = {"Windows": "win", "Darwin": "mac", "Linux": "linux"}
    # 架构缩写映射
    _ARCH_MAP = {
        "amd64": "x64", "x86_64": "x64",
        "arm64": "arm64", "aarch64": "arm64",
        "x86": "x86", "i386": "x86", "i686": "x86",
    }

    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        self.os_type = platform.system()
        self.path_sep = ";" if self.os_type == "Windows" else ":"

        platform_tag = self._PLATFORM_MAP.get(self.os_type, self.os_type.lower())
        arch_raw = platform.machine().lower()
        arch_tag = self._ARCH_MAP.get(arch_raw, arch_raw)
        # date_tag = datetime.now().strftime("%Y%m%d")
        # _{date_tag}
        self.APP_NAME = f"{self._BASE_APP_NAME}_{platform_tag}_{arch_tag}"

        # 动态加载项目根目录的 version.py
        self._ver = self._load_version()

    def _load_version(self):
        version_file = self.project_root / "version.py"
        spec = importlib.util.spec_from_file_location("version", version_file)
        if spec is None or spec.loader is None:
            self.log("ERROR", f"无法加载 version.py: {version_file}")
            sys.exit(1)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def log(self, level: str, message: str):
        print(f"[{level}] {message}")

    def _build_options(self) -> argparse.Namespace:
        """构造传给 flet_cli pack Command 的 options 对象"""
        entry_file = self.project_root / "gui" / "app.py"
        if not entry_file.exists():
            self.log("ERROR", f"入口文件不存在：{entry_file}")
            sys.exit(1)

        root = self.project_root

        # --add-data: action="append", nargs="*" → [[val, ...], ...]
        add_data = [[
            f"{root / 'bootstrap.py'}{self.path_sep}.",
            f"{root / 'core'}{self.path_sep}core",
            f"{root / 'middleware'}{self.path_sep}middleware",
            f"{root / 'gui'}{self.path_sep}gui",
            f"{root / 'utils'}{self.path_sep}utils",
            f"{root / 'builtin'}{self.path_sep}builtin",
            f"{root / '3rd'}{self.path_sep}3rd",
            f"{root / 'assets'}{self.path_sep}assets",
            f"{root / 'resources' / 'bin'}{self.path_sep}resources/bin",
        ]]

        # --hidden-import: action="append", nargs="*" → [[val, ...], ...]
        hidden_imports = [
            "aiosqlite", "sqlite3", "pydantic", "typer",
            "tiktoken.registry", "tiktoken.model", "tiktoken.core",
            "tiktoken_ext", "tiktoken_ext.openai_public",
            "anthropic", "litellm", "jieba", "version",  # 添加 version 模块
            "weixin_sdk", "weixin_sdk.auth", "weixin_sdk.auth.qr_login",
            "weixin_sdk.client", "weixin_sdk.exceptions",
            # SSL and HTTPS related imports
            "ssl", "urllib3", "requests", "httpx", "aiohttp",
            "_ssl", "socket", "select", "selectors",
        ]
        if self.os_type == "Windows":
            hidden_imports += [
                "_overlapped", "_winapi",
                "asyncio.windows_events", "asyncio.windows_utils",
            ]
        hidden_import = [hidden_imports]

        # --pyinstaller-build-args: action="append", nargs="*" → [[val, ...], ...]
        import litellm
        litellm_path = Path(litellm.__file__).parent
        endpoints_json = litellm_path / "containers" / "endpoints.json"

        pyinstaller_build_args = [[
            f"--workpath={root / 'dist' / 'memento_s_build'}",
            "--clean",
            f"--paths={root / '3rd'}",
            "--collect-all=tiktoken",
            "--collect-data=tiktoken",
            "--collect-all=tiktoken_ext",
            "--collect-data=tiktoken_ext",
            "--collect-all=litellm",
            "--collect-all=anthropic",
            "--collect-all=sqlite_vec",
            "--collect-all=tiktoken",
            "--collect-all=flet_desktop",
            "--collect-all=jsonschema_specifications",
            "--collect-all=httpx",
            "--collect-all=aiohttp",
            "--collect-all=requests",
            "--collect-all=urllib3",
            "--collect-submodules=weixin_sdk",
            "--collect-data=weixin_sdk",
            "--collect-all=qrcode",
            "--collect-all=PIL",
            f"--add-data={endpoints_json}{self.path_sep}litellm/containers",
            "--exclude-module=matplotlib",
            "--exclude-module=scipy",
            "--exclude-module=pandas",
            "--exclude-module=jupyter",
            "--exclude-module=notebook",
            "--exclude-module=IPython",
        ]]

        # 图标
        icon = None
        for ext in ("ico", "png", "icns"):
            icon_path = root / "assets" / f"icon.{ext}"
            if icon_path.exists():
                icon = str(icon_path)
                self.log("INFO", f"使用图标: {icon_path}")
                break

        options = argparse.Namespace(
            script=str(entry_file),
            name=self.APP_NAME,
            product_name=self.PRODUCT_NAME,
            distpath=str(root / "dist" / "memento_s_pack"),
            add_data=add_data,
            add_binary=None,
            hidden_import=hidden_import,
            pyinstaller_build_args=pyinstaller_build_args,
            icon=icon,
            onedir=False,
            debug_console=False,
            uac_admin=False,
            codesign_identity=None,
            bundle_id=None,
            non_interactive=True,
            # Windows metadata 从 version.py 动态读取
            product_version=self._ver.__version__,
            file_version=self._ver.__version__ + ".0" if self._ver.__version__.count(".") == 2 else self._ver.__version__,
            file_description=self._ver.file_description,
            company_name=self._ver.company_name,
            copyright=self._ver.copyright,
            verbose=0,
        )
        return options

    def _remove_stale_spec(self):
        """Remove stale generated spec so PyInstaller rebuilds from current options."""
        spec_file = self.project_root / f"{self.APP_NAME}.spec"
        if spec_file.exists():
            spec_file.unlink()
            self.log("INFO", f"删除旧 spec 文件: {spec_file}")

    def build(self) -> bool:
        self.log("INFO", f"开始 memento_s pack 打包（{self.os_type}）...")
        self.log("INFO", f"项目根目录: {self.project_root}")

        try:
            from flet_cli.commands.pack import Command
        except ImportError:
            self.log("ERROR", "flet_cli 未安装，请确认 flet 已正确安装")
            return False

        options = self._build_options()
        self._remove_stale_spec()

        self.log("INFO", "PyInstaller 额外参数:")
        for arg in options.pyinstaller_build_args[0]:
            self.log("INFO", f"  {arg}")
        print()

        try:
            cmd = Command(argparse.ArgumentParser())
            cmd.handle(options)
        except SystemExit as e:
            if e.code == 0:
                dist_dir = self.project_root / "dist" / "memento_s_pack"
                self.log("SUCCESS", f"打包成功！输出目录: {dist_dir}")
                self._print_output_info(dist_dir)
                return True
            else:
                self.log("ERROR", f"打包失败（exit code {e.code}）")
                return False
        except Exception as e:
            self.log("ERROR", f"打包异常: {e}")
            raise

        dist_dir = self.project_root / "dist" / "memento_s_pack"
        self.log("SUCCESS", f"打包成功！输出目录: {dist_dir}")
        self._print_output_info(dist_dir)
        return True

    def _print_output_info(self, dist_dir: Path):
        """打印输出文件信息"""
        candidates = {
            "Windows": [f"{self.APP_NAME}.exe"],
            "Darwin":  [f"{self.APP_NAME}.app", self.APP_NAME],
            "Linux":   [self.APP_NAME],
        }
        for name in candidates.get(self.os_type, [self.APP_NAME]):
            target = dist_dir / name
            if target.exists():
                if target.is_file():
                    size_mb = target.stat().st_size / (1024 * 1024)
                    self.log("INFO", f"文件大小: {size_mb:.2f} MB")
                self.log("INFO", f"路径: {target}")
                break


def main():
    builder = MementoSPackBuilder()
    success = builder.build()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
