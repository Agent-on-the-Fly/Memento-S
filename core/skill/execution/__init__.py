# SPDX-License-Identifier: Apache-2.0
"""execution — Skill 执行基础设施"""

from .agent import SkillAgent
from .adapter import SkillToolAdapter
from .artifact_registry import ArtifactRegistry, ArtifactRecord
from .state import (
    ReActState,
    SkillContext,
    action_signature,
    infer_preferred_extension,
    state_fingerprint,
)
from .result_cache import ResultCache
from .loop_detector import LoopDetector
from .content_analyzer import InfoSaturationDetector
from .error_recovery import StatefulErrorPatternDetector
from .tool_context import RuntimeToolContext, ToolContext
from .tool_args_processor import ToolArgsProcessor
from .tool_result_processor import ToolResultProcessor
from .policy.types import PolicyDecision, PolicyStage
from .policy.pre_execute import run_pre_execute_gate

__all__ = [
    # Core
    "SkillAgent",
    "SkillToolAdapter",
    # Artifact tracking
    "ArtifactRegistry",
    "ArtifactRecord",
    # State
    "ReActState",
    "SkillContext",
    "ResultCache",
    # Detection
    "LoopDetector",
    "InfoSaturationDetector",
    "StatefulErrorPatternDetector",
    # Tool context
    "RuntimeToolContext",
    "ToolContext",  # backward-compat alias
    "ToolArgsProcessor",
    "ToolResultProcessor",
    # Policy
    "PolicyDecision",
    "PolicyStage",
    "run_pre_execute_gate",
    # Helpers
    "action_signature",
    "infer_preferred_extension",
    "state_fingerprint",
]
