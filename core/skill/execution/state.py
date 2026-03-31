from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any


@dataclass
class ReActState:
    query: str
    params: dict[str, Any] | None = None
    max_turns: int = 10
    preferred_core_extension: str | None = None

    # Scratchpad: 持久备忘录，不会被历史压缩影响
    # 模型通过 update_scratchpad 工具写入关键约束/子目标
    # 每轮注入 system prompt 顶部，防止长任务遗忘
    scratchpad: str = ""

    core_artifacts: dict[str, str] = field(default_factory=dict)
    all_artifacts: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    updated_files: list[str] = field(default_factory=list)
    installed_deps: list[str] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    seen_entities: set[str] = field(default_factory=set)
    observation_log: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

    turn_count: int = 0
    tool_calls_count: int = 0
    last_error: str | None = None
    last_error_hash: str | None = None
    last_action_signature: str | None = None
    repeated_action_count: int = 0
    no_progress_count: int = 0
    max_repeated_actions: int = 2
    max_no_progress: int = 2
    last_state_fingerprint: str | None = None
    repeated_state_fingerprint_count: int = 0
    max_repeated_state_fingerprint: int = 2

    # Error tracking for pattern detection
    error_history: list[dict[str, Any]] = field(default_factory=list)
    repeated_error_count: int = 0
    last_recovery_hint_turn: int = 0

    CORE_ARTIFACT_EXTENSIONS: set[str] = field(
        default_factory=lambda: {
            ".pptx",
            ".docx",
            ".xlsx",
            ".pdf",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".rs",
            ".html",
            ".css",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".db",
            ".sqlite",
        }
    )

    def is_core_artifact(self, path: str) -> bool:
        return Path(path).suffix.lower() in self.CORE_ARTIFACT_EXTENSIONS

    def get_primary_artifact(self) -> str | None:
        if (
            self.preferred_core_extension
            and self.preferred_core_extension in self.core_artifacts
        ):
            return self.core_artifacts[self.preferred_core_extension]
        return next(iter(self.core_artifacts.values()), None)

    def _is_similar_filename(self, name1: str, name2: str) -> bool:
        base1 = re.sub(
            r"[_-]?(v\d+|final|draft|copy|backup|new|old)[_-]?", "", name1.lower()
        )
        base2 = re.sub(
            r"[_-]?(v\d+|final|draft|copy|backup|new|old)[_-]?", "", name2.lower()
        )
        return Path(base1).stem == Path(base2).stem

    def lock_artifact(self, path: str) -> tuple[bool, str | None]:
        path_name = Path(path).name
        suffix = Path(path).suffix.lower()

        if path in self.all_artifacts:
            return True, None

        self.all_artifacts.append(path)

        if not self.is_core_artifact(path):
            return True, None

        if self.preferred_core_extension and suffix != self.preferred_core_extension:
            if self.preferred_core_extension in self.core_artifacts:
                preferred = self.core_artifacts[self.preferred_core_extension]
                return (
                    False,
                    f"Primary artifact is locked to {preferred}. Avoid creating new core artifact type '{suffix}'.",
                )

        if suffix in self.core_artifacts:
            existing = self.core_artifacts[suffix]
            existing_name = Path(existing).name
            if self._is_similar_filename(path_name, existing_name):
                return (
                    False,
                    f"Similar core artifact already exists: {existing_name}. Edit that file instead of creating a new version.",
                )
            return True, None

        self.core_artifacts[suffix] = path
        return True, None

    def update_from_observation(self, observation: dict[str, Any]) -> None:
        self.observation_log.append(observation)

        delta = observation.get("state_delta") or {}
        for p in delta.get("created_files", []):
            if p not in self.created_files:
                self.created_files.append(p)
        for p in delta.get("updated_files", []):
            if p not in self.updated_files:
                self.updated_files.append(p)
        for d in delta.get("installed_deps", []):
            if d not in self.installed_deps:
                self.installed_deps.append(d)

    def build_outcome_projection(self) -> dict[str, Any]:
        success_count = sum(
            1 for o in self.observation_log if o.get("exec_status") == "success"
        )
        error_count = sum(
            1 for o in self.observation_log if o.get("exec_status") == "error"
        )

        primary = self.get_primary_artifact()

        return {
            "turn_count": self.turn_count,
            "tool_calls": self.tool_calls_count,
            "primary_artifact": primary,
            "created_files": self.created_files,
            "updated_files": self.updated_files,
            "installed_deps": self.installed_deps,
            "observation_stats": {
                "total": len(self.observation_log),
                "success": success_count,
                "error": error_count,
            },
            "recent_observations": self.observation_log[-5:],
        }

    def update_scratchpad(self, content: str) -> None:
        """更新备忘录内容（追加模式，带时间戳）"""
        entry = f"[Turn {self.turn_count}] {content.strip()}"
        if self.scratchpad:
            self.scratchpad = f"{self.scratchpad}\n{entry}"
        else:
            self.scratchpad = entry
        # 防止无限增长，保留最近 2000 字符
        if len(self.scratchpad) > 2000:
            self.scratchpad = self.scratchpad[-2000:]

    def build_progress_projection(self) -> str:
        created = [Path(p).name for p in self.created_files[-5:]]
        updated = [Path(p).name for p in self.updated_files[-5:]]
        deps = self.installed_deps[-5:]
        primary = self.get_primary_artifact() or "<none>"

        projection = (
            "## Current Execution Progress\n"
            f"- Turns: {self.turn_count}\n"
            f"- Tool calls: {self.tool_calls_count}\n"
            f"- Primary artifact: {primary}\n"
            f"- Created files: {created}\n"
            f"- Updated files: {updated}\n"
            f"- Installed deps: {deps}\n"
        )

        # Scratchpad 注入
        if self.scratchpad:
            projection += (
                "\n## Scratchpad (persistent notes - NEVER ignore these)\n"
                f"{self.scratchpad}\n"
            )

        return projection

    # ========================================================================
    # Error Tracking Methods
    # ========================================================================

    def record_error(
        self, error: str, tool_name: str, hint_injected: bool = False
    ) -> None:
        """Record an error for pattern detection."""
        error_record = {
            "turn": self.turn_count,
            "tool": tool_name,
            "error": error[:500],  # Truncate for storage
            "timestamp": time.time(),
            "was_recovery_hint_injected": hint_injected,
        }
        self.error_history.append(error_record)
        # Keep only last 20 errors
        self.error_history = self.error_history[-20:]

        # Update error hash and count
        current_hash = self._compute_error_fingerprint(error)
        if current_hash == self.last_error_hash:
            self.repeated_error_count += 1
        else:
            self.repeated_error_count = 0
            self.last_error_hash = current_hash

    def _compute_error_fingerprint(self, error: str) -> str | None:
        """Generate normalized error fingerprint."""
        if not error:
            return None

        # Normalize
        normalized = error.lower()
        normalized = re.sub(r"line\s+\d+", "line <n>", normalized)
        normalized = re.sub(r"/[^\s:\"']+", "<path>", normalized)
        normalized = re.sub(r"'[^']+'", "'<var>'", normalized)
        normalized = re.sub(r'"[^"]+"', '"<str>"', normalized)
        normalized = re.sub(r"\b\d+\b", "<n>", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return sha1(normalized.encode("utf-8")).hexdigest()[:16]

    def should_inject_recovery_hint(self, min_interval: int = 2) -> bool:
        """Check if we should inject a recovery hint."""
        return (self.turn_count - self.last_recovery_hint_turn) >= min_interval

    def mark_recovery_hint_injected(self) -> None:
        """Mark that a recovery hint was injected this turn."""
        self.last_recovery_hint_turn = self.turn_count
        if self.error_history:
            self.error_history[-1]["was_recovery_hint_injected"] = True


def infer_preferred_extension(query: str, params: dict[str, Any] | None) -> str | None:
    text = (query or "").lower()
    if params:
        try:
            text += "\n" + json.dumps(params, ensure_ascii=False).lower()
        except Exception:
            text += "\n" + str(params).lower()

    extension_hints = [
        (".pptx", ["pptx", "ppt", "slides", "presentation"]),
        (".docx", ["docx", "word", "document"]),
        (".xlsx", ["xlsx", "excel", "spreadsheet"]),
        (".pdf", ["pdf"]),
        (".md", ["markdown", ".md", "readme"]),
        (".py", ["python", ".py", "script"]),
    ]
    for ext, keywords in extension_hints:
        if any(k in text for k in keywords):
            return ext
    return None


def action_signature(tool_name: str, arguments: Any) -> str:
    if isinstance(arguments, dict):
        normalized = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    else:
        normalized = str(arguments)
    sig_raw = f"{tool_name}|{normalized}"
    return sha1(sig_raw.encode("utf-8")).hexdigest()


def state_fingerprint(observation: dict[str, Any]) -> str:
    delta = observation.get("state_delta") or {}
    payload = {
        "tool": observation.get("tool"),
        "task_signal": observation.get("task_signal", "none"),
        "created_files": sorted(set(delta.get("created_files") or [])),
        "updated_files": sorted(set(delta.get("updated_files") or [])),
        "installed_deps": sorted(set(delta.get("installed_deps") or [])),
        "result_entities": sorted(set(delta.get("result_entities") or [])),
        "error_kind": (observation.get("raw") or {}).get("error_type"),
    }
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return sha1(normalized.encode("utf-8")).hexdigest()
