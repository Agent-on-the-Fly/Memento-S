"""Policy system for Memento-S Agent.

Usage:
    from core.memento_s.policies import PolicyManager

    # Policies are auto-registered on initialization
    pm = PolicyManager()
    result = pm.check("bash_tool", {"command": "ls -la"})
"""

from .base import PolicyFunc, PolicyManager, PolicyResult

__all__ = ["PolicyFunc", "PolicyManager", "PolicyResult"]
