"""embedding utils."""

from __future__ import annotations

import struct
from typing import List


def serialize_f32(vec: List[float]) -> bytes:
    """将 float list 序列化为 little-endian float32 bytes（sqlite-vec 格式）"""
    return struct.pack(f"<{len(vec)}f", *vec)
