
from .protocol import (
    SkillExecuteResult,
    SkillInfo,
    SkillProvider,
    SkillResolveResult,
)

__all__ = [
    "SkillProvider",
    "SkillInfo",
    "SkillExecuteResult",
    "SkillResolveResult",
]

try:
    from .delta_skill_provider import DeltaSkillsProvider
    __all__ = [*__all__, "DeltaSkillsProvider"]
except ImportError:
    DeltaSkillsProvider = None  # type: ignore[misc, assignment]
