"""Filesystem executor aliases with requested naming."""

from .fs import execute_filesystem_op, execute_filesystem_ops, filesystem_tree

__all__ = [
    "filesystem_tree",
    "execute_filesystem_op",
    "execute_filesystem_ops",
]
