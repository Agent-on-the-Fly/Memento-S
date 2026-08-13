# SPDX-License-Identifier: Apache-2.0
"""Skill tool schemas and unified execution gateway.

Skill tool schemas define the function-calling interface exposed to the LLM agent.
SkillDispatcher routes all skill/tool calls through consistent logging.

Implementation is delegated to handlers in skill_dispatch/ to keep each
responsibility isolated and testable.
"""

from __future__ import annotations

import time
import json
from typing import Any, Callable

from core.skill.gateway import SkillGateway
from utils.debug_logger import log_tool_start, log_tool_end
from utils.logger import get_logger

from . import (
    AGENT_TOOL_SCHEMAS,
    TOOL_CREATE_SKILL,
    TOOL_DOWNLOAD_SKILL,
    TOOL_EXECUTE_SKILL,
    TOOL_RECALL_CONTEXT,
    TOOL_SEARCH_SKILL,
    ContextRecallHandler,
    CreatedSkillInstaller,
    HallucinationGuard,
    SkillExecutionHandler,
    SkillSearchHandler,
)

logger = get_logger(__name__)

SkillsChangedCallback = Callable[[], None]

_WORKSPACE_MARKER = "/memento_s/workspace/"


def _normalize_skill_request(request: str) -> str:
    """Strip session workspace absolute-path prefix from skill_request, keep filename only.

    Planner LLM has no knowledge of session directory structure. When a user's
    original request contains an absolute path under the session workspace,
    the planner copies it verbatim into skill_request. This helper removes
    the session workspace prefix wherever it appears, so the skill agent receives
    just the filename (e.g. "output.pptx") instead of the full absolute path.

    Example: "... /Users/xxx/memento_s/workspace/output/sam3.pptx" → "... sam3.pptx"

    Absolute paths outside the session workspace are preserved intact.
    """
    idx = request.find(_WORKSPACE_MARKER)
    if idx == -1:
        return request
    # find() returns first match — search backward to the / before username dir
    slash_start = request.rfind("/", 0, idx)
    if slash_start == -1:
        return request
    after = request[idx + len(_WORKSPACE_MARKER):]
    segments = [s for s in after.split("/") if s.strip()]
    if not segments:
        return request
    return segments[-1]


class SkillDispatcher:
    """Unified entry point for executing agent-exposed skill tools.

    Handles:
    - search_skill / execute_skill / download_skill / create_skill via SkillGateway

    Implementation delegates to handlers in skill_dispatch/:
    - SkillSearchHandler: search local + cloud skills
    - SkillExecutionHandler: execute / download / create skills
    - HallucinationGuard: validates hallucinated skill names
    - ContextRecallHandler: recall_context tool
    - CreatedSkillInstaller: post skill-creator install logic
    """

    def __init__(
        self,
        skill_gateway: SkillGateway,
        on_skills_changed: SkillsChangedCallback | None = None,
        on_skill_step: Any | None = None,
        infra: Any | None = None,
    ):
        self._gateway = skill_gateway
        self._ctx: Any | None = None
        self._on_skills_changed = on_skills_changed
        self._on_skill_step = on_skill_step
        self._infra = infra
        self._step_summary_source: Callable[[], str] | None = None

        # ── Initialise handlers ──────────────────────────────────────────────
        self._search = SkillSearchHandler(skill_gateway)
        self._recall = ContextRecallHandler(infra, None)
        self._hallucination = HallucinationGuard(skill_gateway)
        self._installer = CreatedSkillInstaller(skill_gateway, None)
        self._execution = SkillExecutionHandler(
            gateway=skill_gateway,
            on_step=on_skill_step,
            step_summary_source=None,  # set via set_step_summary_source
            on_skills_changed=on_skills_changed,
        )
        # Wire skill-creator success → post-install
        self._execution.set_on_skill_created(self._installer.install_from_workspace)

        # ── Load schemas ────────────────────────────────────────────────────
        from tools import get_tool_schemas

        schemas = get_tool_schemas(category="skill")
        self._skill_schemas = schemas if schemas else AGENT_TOOL_SCHEMAS

    def get_skill_tool_schemas(self) -> list[dict]:
        """Return skill tool schemas loaded from tools registry."""
        return self._skill_schemas

    @property
    def max_evolution_updates_per_step(self) -> int:
        return max(0, int(self._gateway._config.evolution_max_updates_per_step))

    def set_context(self, ctx: Any) -> None:
        """Set SessionContext for this dispatcher (created by agent layer)."""
        self._ctx = ctx
        session_id = ctx.session_id if ctx else None
        self._recall = ContextRecallHandler(self._infra, session_id)
        self._installer = CreatedSkillInstaller(self._gateway, session_id)
        self._execution.set_session_id(session_id)

    def set_on_skills_changed(self, callback: SkillsChangedCallback | None) -> None:
        """Set callback invoked after create_skill / download_skill succeeds."""
        self._on_skills_changed = callback
        self._execution._on_skills_changed = callback

    def set_on_skill_step(self, callback: Any | None) -> None:
        """Set callback invoked during skill execution for each step."""
        self._on_skill_step = callback
        self._execution._on_step = callback

    def set_step_summary_source(self, source: Callable[[], str] | None) -> None:
        """Set callable that returns the last step's structured summary."""
        self._step_summary_source = source
        self._execution._step_summary_source = source

    async def execute(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute an agent-exposed tool by name."""
        start_time = time.monotonic()
        call_id = f"{tool_name}_{int(start_time * 1000)}"

        log_tool_start(tool_name, args, call_id)

        try:
            if tool_name == TOOL_SEARCH_SKILL:
                result = await self._search.search(args)
            elif tool_name == TOOL_EXECUTE_SKILL:
                if "request" in args:
                    original = args["request"]
                    normalized = _normalize_skill_request(original)
                    if normalized != original:
                        args = dict(args)
                        args["request"] = normalized
                result = await self._execution.execute_skill(args)
            elif tool_name == TOOL_DOWNLOAD_SKILL:
                result = await self._execution.download_skill(args)
            elif tool_name == TOOL_CREATE_SKILL:
                result = await self._execution.create_skill(args)
            elif tool_name == TOOL_RECALL_CONTEXT:
                result = await self._recall.recall(args)
            elif tool_name == "skill_list":
                manifests = await self._gateway.discover()
                verbose = bool(args.get("verbose", False))
                output = []
                for manifest in manifests:
                    item = {
                        "name": manifest.name,
                        "description": manifest.description,
                    }
                    if verbose:
                        item.update(
                            {
                                "execution_mode": manifest.execution_mode.value,
                                "parameters": manifest.parameters,
                                "dependencies": manifest.dependencies,
                                "governance": manifest.governance.model_dump(mode="json"),
                            }
                        )
                    output.append(item)
                result = json.dumps(
                    {"ok": True, "status": "success", "output": output},
                    ensure_ascii=False,
                )
            elif tool_name == "read_skill":
                skill_name = str(args.get("skill_name", "")).strip()
                if not skill_name:
                    result = json.dumps(
                        {
                            "ok": False,
                            "status": "failed",
                            "error_code": "INVALID_INPUT",
                            "summary": "skill_name is required for read_skill",
                        }
                    )
                else:
                    content = await self._gateway.read(skill_name)
                    result = json.dumps(
                        {
                            "ok": content is not None,
                            "status": "success" if content is not None else "failed",
                            "error_code": None if content is not None else "SKILL_NOT_FOUND",
                            "skill_name": skill_name,
                            "output": content,
                        },
                        ensure_ascii=False,
                    )
            elif tool_name == "skill_install":
                result = await self._execution.download_skill(args)
            elif tool_name == "bash_tool":
                command = str(args.get("command", ""))
                if "rm -rf /" in command:
                    raise PermissionError("dangerous root deletion command denied")
                raise ValueError("bash_tool is not exposed by SkillDispatcher")
            else:
                # Hallucination Interceptor: Auto-convert skill name to execute_skill
                result = await self._hallucination.intercept(
                    tool_name, args, self._execution.execute_skill
                )

            duration = time.monotonic() - start_time
            log_tool_end(tool_name, result, duration, success=True)
            return result

        except Exception as e:
            duration = time.monotonic() - start_time
            error_result = json.dumps({"ok": False, "error": str(e)})
            log_tool_end(tool_name, error_result, duration, success=False)
            raise

    async def evolve_after_failure(
        self,
        *,
        task: str,
        trace: list[dict[str, Any]],
        rationale: str,
    ) -> dict[str, Any]:
        """Run the gateway's guarded Read-Write update for a failed trace."""
        used_skills = [
            str(item.get("skill_name", ""))
            for item in trace
            if item.get("skill_name")
        ]
        result = await self._gateway.evolve_failure(
            task=task,
            used_skills=used_skills,
            trace=trace,
            rationale=rationale,
            session_id=self._ctx.session_id if self._ctx else "",
        )
        if result.deployed:
            self._notify_skills_changed()
        return result.to_dict()

    def _notify_skills_changed(self) -> None:
        if self._on_skills_changed is not None:
            try:
                self._on_skills_changed()
            except Exception:
                logger.opt(exception=True).warning("on_skills_changed callback failed")
