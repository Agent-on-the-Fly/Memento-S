"""embedding — 向量化基础模块（共享）"""

from .client import (
    EmbeddingClient,
    EmbeddingClientConfig,
    EmbeddingError,
    EmbeddingAPIError,
    EmbeddingConfigError,
    EmbeddingStats,
)
from .utils import serialize_f32

__all__ = [
    "EmbeddingClient",
    "EmbeddingClientConfig",
    "EmbeddingError",
    "EmbeddingAPIError",
    "EmbeddingConfigError",
    "EmbeddingStats",
    "serialize_f32",
]
