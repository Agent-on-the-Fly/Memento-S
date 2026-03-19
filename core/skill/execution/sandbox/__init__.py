"""sandbox — 沙箱执行（本地子进程 / E2B 云端 / uv 虚拟环境）"""

from .base import BaseSandbox, get_sandbox
from .uv import UvLocalSandbox

__all__ = [
    "BaseSandbox",
    "get_sandbox",
    "UvLocalSandbox",
]
