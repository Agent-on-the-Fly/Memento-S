# SPDX-License-Identifier: Apache-2.0
"""Skill execution handler — executes local skills, handles download/create delegation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.skill.gateway import SkillGateway

logger = __import__("utils.logger", fromlist=["get_logger"]).get_logger(__name__)


class SkillExecutionHandler:
    """Handles execute_skill, download_skill, and create_skill tool calls.

    download_skill and create_skill are delegated here (not routed through gateway.execute)
    so they can trigger the skills-changed callback.
    """

    def __init__(
        self,
        gateway: SkillGateway,
        on_step: Any | None,
        step_summary_source: Callable[[], str] | None,
        on_skills_changed: Callable[[], None] | None,
    ) -> None:
        self._gateway = gateway
        self._session_id: str | None = None
        self._on_step = on_step
        self._step_summary_source = step_summary_source
        self._on_skills_changed = on_skills_changed
        self._on_skill_created: Callable[[], Any] | None = None

    def set_session_id(self, session_id: str | None) -> None:
        self._session_id = session_id

    def set_on_skill_created(self, callback: Callable[[], Any]) -> None:
        """Set callback invoked after skill-creator succeeds.

        Wraps the provided callback so it also triggers _notify_skills_changed.
        """
        async def wrapped() -> None:
            await callback()
            self._notify_skills_changed()

        self._on_skill_created = wrapped

    # ── execute_skill ───────────────────────────────────────────────────────

    async def execute_skill(self, args: dict[str, Any]) -> str:
        """Execute a local installed skill via gateway."""
        args = dict(args)
        skill_name = args.pop("skill_name", "").strip("> \t\n")
        from utils.log_config import log_preview, log_preview_long
        from utils.debug_logger import log_agent_phase

        log_agent_phase(
            "SKILL_EXECUTE", skill_name,
            f"request='{log_preview_long(args.get('request', ''))}'"
        )
        logger.info(
            "SkillExecutionHandler.execute_skill: skill_name={}, query_preview={}",
            skill_name,
            log_preview(args.get("request", ""), default=200),
        )

        if not skill_name:
            return json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error_code": "INVALID_INPUT",
                    "summary": "skill_name is required for execute_skill",
                },
                ensure_ascii=False,
                default=str,
            )

        if self._step_summary_source:
            prior = self._step_summary_source()
            if prior:
                args["prior_context"] = prior

        envelope = await self._gateway.execute(
            skill_name=skill_name,
            params=args,
            session_id=self._session_id,
            on_step=self._on_step,
        )
        await self._gateway.record_skill_outcome(skill_name, envelope.ok)

        logger.debug(
            f"[SkillExecutionHandler.execute_skill] gateway.execute RETURNED: "
            f"skill_name={skill_name}, ok={envelope.ok}, "
            f"status={envelope.status}, summary='{envelope.summary or ''}'"
        )

        _out = envelope.output or {}
        _cf = list(envelope.artifacts) if envelope.artifacts else []
        _uf: list = []

        logger.info(
            "[ANALYSIS-LOG] skill_result: skill={}, ok={}, created_files={}, "
            "updated_files={}",
            skill_name, envelope.ok, _cf, _uf,
        )

        # After skill-creator succeeds, trigger post-install callback
        if envelope.ok and skill_name == "skill-creator" and self._on_skill_created:
            await self._on_skill_created()

        payload: dict[str, Any] = {
            "ok": envelope.ok,
            "status": envelope.status.value,
            "summary": envelope.summary,
            "skill_name": envelope.skill_name,
            "output": envelope.output,
        }
        if envelope.error_code:
            payload["error_code"] = envelope.error_code.value
        if envelope.outputs:
            payload["outputs"] = envelope.outputs
        if envelope.artifacts:
            payload["artifacts"] = envelope.artifacts
        if envelope.diagnostics:
            payload["diagnostics"] = envelope.diagnostics
        return json.dumps(payload, ensure_ascii=False, default=str)

    # ── download_skill ──────────────────────────────────────────────────────

    async def download_skill(self, args: dict[str, Any]) -> str:
        """Download a cloud skill to local storage and notify skills changed."""
        skill_name = str(args.get("skill_name", "")).strip()

        if not skill_name:
            return json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error_code": "INVALID_INPUT",
                    "summary": "skill_name is required for download_skill",
                },
                ensure_ascii=False,
                default=str,
            )

        try:
            skill = await self._gateway.install(skill_name)
            if skill:
                self._notify_skills_changed()
                return json.dumps(
                    {
                        "ok": True,
                        "status": "success",
                        "summary": f"Skill '{skill_name}' downloaded and installed successfully.",
                        "skill_name": skill.name,
                        "output": f"Skill '{skill_name}' is now installed and ready to use via `execute_skill(skill_name='{skill_name}', request='...')`",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            else:
                return json.dumps(
                    {
                        "ok": False,
                        "status": "failed",
                        "error_code": "DOWNLOAD_FAILED",
                        "summary": f"Failed to download skill '{skill_name}'. It may not exist in the cloud or the download failed.",
                    },
                    ensure_ascii=False,
                    default=str,
                )
        except Exception as e:
            logger.exception("Failed to download skill")
            return json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error_code": "INTERNAL_ERROR",
                    "summary": f"Error downloading skill: {str(e)}",
                },
                ensure_ascii=False,
                default=str,
            )

    # ── create_skill (delegates to skill-creator) ──────────────────────────

    async def create_skill(self, args: dict[str, Any]) -> str:
        """Create a new skill by delegating to skill-creator."""
        request = str(args.get("request", "")).strip()

        if not request:
            return json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error_code": "INVALID_INPUT",
                    "summary": "request is required for create_skill - describe what skill you want to create",
                },
                ensure_ascii=False,
                default=str,
            )

        execute_args = {
            "skill_name": "skill-creator",
            "request": request,
        }

        logger.info(
            "SkillExecutionHandler: delegating create_skill to skill-creator: request_preview={}",
            request[:100],
        )
        return await self.execute_skill(execute_args)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _notify_skills_changed(self) -> None:
        """Invoke the skills-changed callback if registered."""
        if self._on_skills_changed is not None:
            try:
                self._on_skills_changed()
            except Exception:
                logger.opt(exception=True).warning("on_skills_changed callback failed")
