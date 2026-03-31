"""retrieval — 技能检索（本地向量 + 远程召回 + 多路合并）"""

from .base import BaseRecall
from .local_file_recall import LocalFileRecall
from .multi_recall import MultiRecall
from .remote_recall import RemoteRecall
from .schema import RecallCandidate

# LocalDbRecall 是可选的（依赖 sqlite-vec）
try:
    from .local_db_recall import LocalDbRecall

    __all__ = [
        "BaseRecall",
        "LocalDbRecall",
        "LocalFileRecall",
        "MultiRecall",
        "RecallCandidate",
        "RemoteRecall",
    ]
except ImportError:
    __all__ = [
        "BaseRecall",
        "LocalFileRecall",
        "MultiRecall",
        "RecallCandidate",
        "RemoteRecall",
    ]
