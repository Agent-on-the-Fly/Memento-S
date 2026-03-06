
from __future__ import annotations

import ast
import sys

from core.config.logging import get_logger
from .parsing import parse_code
from ...schema import Skill

logger = get_logger(__name__)

_IMPORT_TO_PYPI: dict[str, str] = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "np": "numpy",
    "pd": "pandas",
    "pptx": "python-pptx",
    "docx": "python-docx",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    "dotenv": "python-dotenv",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "websocket": "websocket-client",
    "Crypto": "pycryptodome",
    "jwt": "PyJWT",
    "nacl": "PyNaCl",
    "psycopg2": "psycopg2-binary",
    "pymongo": "pymongo",
    "redis": "redis",
    "sqlalchemy": "sqlalchemy",
    "dateutil": "python-dateutil",
    "tqdm": "tqdm",
    "rich": "rich",
    "click": "click",
    "attr": "attrs",
    "pydantic": "pydantic",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "seaborn": "seaborn",
    "defusedxml": "defusedxml",
    "six": "six",
    "chardet": "chardet",
    "colorama": "colorama",
}

_STDLIB: set[str] = set()
if hasattr(sys, "stdlib_module_names"):
    _STDLIB = set(sys.stdlib_module_names)
else:
    _STDLIB = {
        "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
        "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
        "binhex", "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb",
        "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections",
        "colorsys", "compileall", "concurrent", "configparser", "contextlib",
        "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv",
        "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal",
        "difflib", "dis", "distutils", "doctest", "email", "encodings",
        "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
        "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt",
        "getpass", "gettext", "glob", "graphlib", "grp", "gzip", "hashlib",
        "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
        "imp", "importlib", "inspect", "io", "ipaddress", "itertools",
        "json", "keyword", "lib2to3", "linecache", "locale", "logging",
        "lzma", "mailbox", "mailcap", "marshal", "math", "mimetypes",
        "mmap", "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
        "numbers", "operator", "optparse", "os", "ossaudiodev", "pathlib",
        "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
        "plistlib", "poplib", "posix", "posixpath", "pprint", "profile",
        "pstats", "pty", "pwd", "py_compile", "pyclbr", "pydoc",
        "queue", "quopri", "random", "re", "readline", "reprlib",
        "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
        "selectors", "shelve", "shlex", "shutil", "signal", "site",
        "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "sqlite3",
        "ssl", "stat", "statistics", "string", "stringprep", "struct",
        "subprocess", "sunau", "symtable", "sys", "sysconfig", "syslog",
        "tabnanny", "tarfile", "telnetlib", "tempfile", "termios", "test",
        "textwrap", "threading", "time", "timeit", "tkinter", "token",
        "tokenize", "tomllib", "trace", "traceback", "tracemalloc",
        "tty", "turtle", "turtledemo", "types", "typing", "unicodedata",
        "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave",
        "weakref", "webbrowser", "winreg", "winsound", "wsgiref",
        "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile", "zipimport",
        "zlib", "_thread",
    }


def _extract_imports(code: str) -> set[str]:
    tree = parse_code(code)
    if tree is None:
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def extract_dependencies(code: str) -> list[str]:
    if not code or not code.strip():
        return []
    raw_imports = _extract_imports(code)
    third_party = {m for m in raw_imports if m not in _STDLIB and not m.startswith("_")}
    pip_names = {_IMPORT_TO_PYPI.get(m, m) for m in third_party}
    result = sorted(pip_names)
    if result:
        logger.debug("Extracted dependencies from code: %s", result)
    return result


def ensure_dependencies(skill: Skill) -> list[str]:
    declared = set(skill.dependencies or [])
    analyzed = set(extract_dependencies(skill.code))
    merged = declared | analyzed
    if analyzed - declared:
        logger.info(
            "Skill '%s': auto-detected additional dependencies: %s",
            skill.name, sorted(analyzed - declared),
        )
    return sorted(merged)
