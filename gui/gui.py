#!/usr/bin/env python3
"""
Memento-S GUI Entry Point
"""

import sys
from pathlib import Path


# === 终极修复：彻底解决 tiktoken 找不到编码的警告 ===
try:
    import tiktoken_ext.openai_public
    import tiktoken_ext.anthropic
except ImportError:
    pass

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bootstrap import bootstrap_sync

from gui.app import main

if __name__ == "__main__":
    main()
