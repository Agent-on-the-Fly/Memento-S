"""core.protocol — public interface for the AG-UI event protocol.

All consumers (GUI, CLI, Feishu) and producers (agent, phases) import
from this package.  No code outside ``core/protocol/`` should import
sub-modules of ``core/memento_s/`` for protocol types.
"""

from .adapter import AGUIProtocolAdapter, ProtocolAdapter
from .events import AGUIEventType, build_event, new_run_id
from .pipeline import (
    AGUIEventPipeline,
    AGUIEventSink,
    PersistenceSink,
    RunAccumulator,
    ToolTranscriptSink,
)
from .run_emitter import RunEmitter
from .types import AgentFinishReason, PhaseSignalType, PlanStepStatus, RunStatus, StepStatus

__all__ = [
    "AGUIEventPipeline",
    "AGUIEventSink",
    "AGUIEventType",
    "AGUIProtocolAdapter",
    "AgentFinishReason",
    "PersistenceSink",
    "PhaseSignalType",
    "PlanStepStatus",
    "ProtocolAdapter",
    "RunAccumulator",
    "RunEmitter",
    "RunStatus",
    "StepStatus",
    "ToolTranscriptSink",
    "build_event",
    "new_run_id",
]
