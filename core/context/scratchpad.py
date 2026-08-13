# SPDX-License-Identifier: Apache-2.0
"""Legacy session scratchpad used by context compatibility APIs.

Large tool results are archived as separate artifacts; explicit notes are
appended to a small markdown index that can be referenced from prompts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class Scratchpad:
    def __init__(self, session_id: str, date_dir: Path) -> None:
        self.session_id = session_id
        self.date_dir = Path(date_dir)
        self.date_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)
        self.path = self.date_dir / f"scratchpad_{safe_id}.md"
        self.artifacts_dir = self.date_dir / f"scratchpad_{safe_id}_artifacts"
        self._section_count = 0
        self._artifact_count = 0
        if not self.path.exists():
            self.path.write_text(
                f"# Session Scratchpad\n\nSession: {session_id}\n",
                encoding="utf-8",
            )

    def write(self, title: str, content: str) -> str:
        self._section_count += 1
        anchor = f"section-{self._section_count}"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n<a id=\"{anchor}\"></a>\n## {title}\n\n{content}\n")
        return f"scratchpad#{anchor}"

    def build_reference(self) -> str:
        if self._section_count == 0:
            return ""
        return (
            "## Scratchpad (archived context)\n\n"
            f"Archived details: `{self.path}`. Use `filesystem` to read it or "
            "`search_grep` to find a specific passage."
        )

    def persist_tool_result(
        self, tool_call_id: str, tool_name: str, result: str
    ) -> dict[str, Any]:
        message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }
        if len(result) <= 4000 and result.count("\n") <= 120:
            return message

        self._artifact_count += 1
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        safe_tool = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name)
        artifact = self.artifacts_dir / (
            f"artifact_{self._artifact_count:04d}_{safe_tool}.txt"
        )
        artifact.write_text(result, encoding="utf-8")

        lines = result.splitlines()
        preview = "\n".join(lines[:5])[:500]
        message["content"] = (
            f"[artifact_ref:{artifact}]\n{preview}\n"
            f"[{len(result)} chars, full content archived]"
        )
        return message

    @staticmethod
    def _format_for_scratchpad(raw: str) -> str:
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return raw
        if not isinstance(data, dict):
            return raw

        results = data.get("results")
        if isinstance(results, list) and results:
            sections: list[str] = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                tool = str(item.get("tool", "tool"))
                args = item.get("args") or {}
                label = (
                    args.get("path")
                    or args.get("query")
                    or args.get("command")
                    or ""
                )
                heading = f"### {tool}: {label}" if label else f"### {tool}"
                if item.get("error") is not None:
                    body = f"**ERROR**: {item['error']}"
                else:
                    body = str(item.get("result", ""))
                sections.append(f"{heading}\n\n{body}")
            return "\n\n".join(sections) if sections else raw

        if "skill_name" in data and ("summary" in data or "output" in data):
            status = "OK" if data.get("ok", False) else "FAIL"
            parts = [f"### **{data['skill_name']}** — {status}"]
            if data.get("summary"):
                parts.append(str(data["summary"]))
            if data.get("output") is not None:
                parts.append(str(data["output"]))
            if data.get("diagnostics") is not None:
                parts.append(
                    "Diagnostics: "
                    + json.dumps(data["diagnostics"], ensure_ascii=False)
                )
            return "\n\n".join(parts)
        return raw
