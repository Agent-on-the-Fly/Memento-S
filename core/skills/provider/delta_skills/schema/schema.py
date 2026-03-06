
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

EXECUTION_MODES = ("function", "knowledge", "playbook")

_KNOWLEDGE_BODY_MAX_CHARS = 800


def _check_is_knowledge(code: str) -> bool:
    if not code or not code.strip():
        return True
    text = code.lstrip()
    if text.startswith("---"):
        fm_end = text.find("\n---", 3)
        if fm_end != -1:
            body = text[fm_end + 4:].strip()
            if not body:
                return True
            try:
                ast.parse(body)
                return False
            except SyntaxError:
                return True
        else:
            pass
    try:
        ast.parse(text)
        return False
    except SyntaxError:
        return True


def _check_is_playbook(source_dir: str | None) -> bool:
    if not source_dir:
        return False
    scripts_dir = Path(source_dir) / "scripts"
    if not scripts_dir.exists():
        return False
    py_files = [
        p for p in scripts_dir.glob("*.py")
        if p.name != "__init__.py"
        and p.stem != "example"          # init_skill 
        and not p.stem.endswith("_test")  # 
        and not p.stem.startswith("test_")
    ]
    return len(py_files) >= 2


class Skill(BaseModel):

    name: str = Field(..., description=" calculate_sum")
    description: str = Field(..., description="")
    code: str = Field(..., description="Python")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Schema")
    dependencies: list[str] = Field(default_factory=list, description="")
    tags: list[str] = Field(default_factory=list, description="")
    version: int = Field(0, description="0 ")
    stable_version: Optional[int] = Field(None, description="")
    files: dict[str, str] = Field(default_factory=dict, description=" {filename: content}")
    source_dir: Optional[str] = Field(None, description="")
    execution_mode: Optional[Literal["function", "knowledge", "playbook"]] = Field(
        None,
        description=" None ",
    )
    entry_script: Optional[str] = Field(
        None,
        description="playbook  .py ",
    )

    _is_knowledge: bool | None = None
    _is_playbook: bool | None = None

    def model_post_init(self, __context: Any) -> None:
        self._is_knowledge = _check_is_knowledge(self.code)
        self._is_playbook = _check_is_playbook(self.source_dir)

    @property
    def is_knowledge_skill(self) -> bool:
        if self.execution_mode is not None:
            return self.execution_mode == "knowledge"
        if self._is_knowledge is None:
            self._is_knowledge = _check_is_knowledge(self.code)
        return self._is_knowledge

    @property
    def is_playbook(self) -> bool:
        if self.execution_mode is not None:
            return self.execution_mode == "playbook"
        if self._is_playbook is None:
            self._is_playbook = _check_is_playbook(self.source_dir)
        return self._is_playbook

    def invalidate_cache(self) -> None:
        self._is_knowledge = _check_is_knowledge(self.code)
        self._is_playbook = _check_is_playbook(self.source_dir)

    def to_embedding_text(self) -> str:
        parts = [self.name.replace("_", " "), self.description]

        if self.is_knowledge_skill:
            body = self._extract_md_body()
            if body:
                parts.append(body)
        else:
            if self.parameters:
                param_names = " ".join(self.parameters.keys())
                parts.append(f"parameters: {param_names}")

            if self.code:
                try:
                    tree = ast.parse(self.code)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            ds = ast.get_docstring(node)
                            if ds:
                                parts.append(ds)
                            break
                except SyntaxError:
                    pass

        if self.dependencies:
            parts.append(f"dependencies: {' '.join(self.dependencies)}")

        if self.tags:
            parts.append(f"tags: {' '.join(self.tags)}")

        return " | ".join(parts)

    def _extract_md_body(self) -> str:
        text = self.code
        if not text:
            return ""

        text = re.sub(r"^---\s*\n.*?\n---\s*\n?", "", text, count=1, flags=re.DOTALL)

        text = re.sub(r"```[\s\S]*?```", " ", text)

        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*]\s+", " ", text, flags=re.MULTILINE)
        text = re.sub(r"\|", " ", text)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"`([^`]*)`", r"\1", text)

        text = re.sub(r"\s+", " ", text).strip()

        return text[:_KNOWLEDGE_BODY_MAX_CHARS]


class SkillExecutionResult(BaseModel):
    success: bool
    result: Any
    error: str | None = None
    skill_name: str
    artifacts: list[str] = []


def get_delta_meta(meta: dict) -> dict:
    raw_metadata = meta.get("metadata")
    if isinstance(raw_metadata, dict):
        delta = raw_metadata.get("delta-skills", {})
        if isinstance(delta, dict) and delta:
            return delta
    return {
        k: meta[k]
        for k in ("function_name", "parameters", "dependencies")
        if k in meta
    }
