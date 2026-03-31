"""Environment variable whitelist filtering."""

from __future__ import annotations

import os

# Environment variable whitelist patterns
# All platform entries merged; missing vars simply won't match.
ENV_WHITELIST_PATTERNS: set[str] = {
    # universal
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "UV_*",
    "MEMENTO_*",
    # pip / PyPI mirror
    "PIP_*",
    # proxy / certs (cross-platform)
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    # POSIX-common (absent on Windows → ignored)
    "HOME",
    "USER",
    "LOGNAME",
    "TERM",
    "SHELL",
    "TMPDIR",
    "XDG_RUNTIME_DIR",
    # Windows-common (absent on POSIX → ignored)
    "USERPROFILE",
    "TEMP",
    "TMP",
    "COMSPEC",
    "PATHEXT",
    "APPDATA",
    "LOCALAPPDATA",
    "SystemRoot",
    "HOMEDRIVE",
    "HOMEPATH",
    "SYSTEMROOT",
    "WINDIR",
    "SYSTEMDRIVE",
}


def filter_env_by_whitelist(
    source: dict[str, str] | None = None,
    whitelist: set[str] | None = None,
) -> dict[str, str]:
    """Filter environment variables through the whitelist.

    Supports trailing ``*`` as a prefix-match wildcard.

    Args:
        source: Source environment variables. Defaults to os.environ.
        whitelist: Custom whitelist patterns. Defaults to ENV_WHITELIST_PATTERNS.

    Returns:
        Filtered environment variables dictionary.
    """
    patterns = whitelist if whitelist is not None else ENV_WHITELIST_PATTERNS
    src = source if source is not None else dict(os.environ)
    return {
        k: v
        for k, v in src.items()
        if any(k == w or (w.endswith("*") and k.startswith(w[:-1])) for w in patterns)
    }
