"""Public entry points for skill-engine helpers.

This package-level module provides a stable import surface for higher layers
(`cli`, `agent.py`) and for contributors extending core behaviour.
"""

from core.skill_engine import skill_runner as _skill_runner

# Re-export the public API defined by skill_runner without duplicating symbol lists.
for _name in _skill_runner.__all__:
    globals()[_name] = getattr(_skill_runner, _name)

__all__ = list(_skill_runner.__all__)
