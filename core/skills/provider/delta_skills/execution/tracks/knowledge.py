
from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.config import g_settings
from core.config.logging import get_logger

from ...schema import Skill, SkillExecutionResult
from .base import BaseTrackExecutor
from .prompts import KNOWLEDGE_EXECUTE_PROMPT
from .utils import detect_skill_mismatch, parse_knowledge_response

logger = get_logger(__name__)

_RE_JSON_BLOCK = re.compile(r"```(?:json)?\s*\n(.*?)\n\s*```", re.DOTALL)

_RE_FULL_CODE_BLOCK = re.compile(
    r"^\s*```(?:python|py)?\s*\n(.*)\n\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)


_RE_BASH_BLOCK = re.compile(
    r"```(?:bash|sh)\s*\n(.*?)\n?\s*```",
    re.DOTALL | re.IGNORECASE,
)

_RE_PYTHON_CMD = re.compile(
    r"^[ \t]*python3?\s+((?:\S+/)\S+\.py(?:\s+\S+)*)\s*$",
    re.MULTILINE,
)


def _resolve_script_path(raw_path: str, source_dir: str | None) -> str:
    p = Path(raw_path)
    if p.is_absolute():
        return raw_path
    if source_dir:
        resolved = (Path(source_dir) / p).resolve()
        return str(resolved)
    return raw_path


def _bash_block_to_ops(block_content: str, source_dir: str | None) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    for line in block_content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _RE_PYTHON_CMD.match(line)
        if m:
            parts = m.group(1).split()
            script_path = _resolve_script_path(parts[0], source_dir)
            extra_args = parts[1:]
            cmd_parts = [sys.executable, script_path] + extra_args
            command = " ".join(cmd_parts)
        else:
            command = stripped
        ops.append({"type": "run_command", "command": command})
    return ops


def _python_cmd_lines_to_ops(skill_code: str, source_dir: str | None) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    for m in _RE_PYTHON_CMD.finditer(skill_code):
        parts = m.group(1).split()
        script_path = _resolve_script_path(parts[0], source_dir)
        extra_args = parts[1:]
        cmd_parts = [sys.executable, script_path] + extra_args
        ops.append({"type": "run_command", "command": " ".join(cmd_parts)})
    return ops


def extract_commands_from_skill_md(
    skill_code: str,
    source_dir: str | None,
) -> list[dict[str, Any]] | None:
    body = skill_code
    fm_match = re.match(r"^---\s*\n.*?\n---\s*\n?", skill_code, re.DOTALL)
    if fm_match:
        body = skill_code[fm_match.end():]

    ops: list[dict[str, Any]] = []

    for block_match in _RE_BASH_BLOCK.finditer(body):
        block_ops = _bash_block_to_ops(block_match.group(1), source_dir)
        ops.extend(block_ops)

    if not ops:
        ops = _python_cmd_lines_to_ops(body, source_dir)

    return ops if ops else None


def _extract_executable_python(text: str) -> str | None:
    m = _RE_FULL_CODE_BLOCK.match(text.strip())
    if not m:
        return None
    code = m.group(1).strip()
    return code if code else None


def _extract_ops_json(text: str) -> list[dict] | None:
    candidates: list[str] = []
    for m in _RE_JSON_BLOCK.finditer(text):
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and isinstance(parsed.get("ops"), list):
                return parsed["ops"]
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _ops_to_python(ops: list[dict], skill_name: str) -> str:
    ops_json = json.dumps(ops, ensure_ascii=False)
    workspace_path = str(g_settings.workspace_path)

    code = f'''\
import subprocess, os, shutil, json
from pathlib import Path

_OPS = json.loads({ops_json!r})
_WORKSPACE = {workspace_path!r}

def _resolve_cwd(raw_cwd):
    """ LLM  working_dir /workspace →  workspace """
    if not raw_cwd or raw_cwd == ".":
        return _WORKSPACE
    p = Path(raw_cwd)
    if not p.exists():
        for prefix in ("/workspace", "/home/user/workspace"):
            if raw_cwd == prefix or raw_cwd.startswith(prefix + "/"):
                relative = raw_cwd[len(prefix):].lstrip("/")
                resolved = Path(_WORKSPACE) / relative if relative else Path(_WORKSPACE)
                resolved.mkdir(parents=True, exist_ok=True)
                return str(resolved)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return raw_cwd
        except OSError:
            return _WORKSPACE
    return raw_cwd

def _run_op(op):
    op_type = op.get("type", "")

    if op_type == "run_command":
        cmd = op.get("command", "")
        cwd = _resolve_cwd(op.get("working_dir", "."))
        timeout = op.get("timeout", 120)
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)
            out = r.stdout.strip()
            err = r.stderr.strip()
            if r.returncode != 0:
                print("[WARN] Command exited with code", r.returncode)
                if err:
                    print("stderr:", err[:2000])
            if out:
                print(out)
        except subprocess.TimeoutExpired:
            print("[ERROR] Command timed out after", timeout, "seconds:", cmd)

    elif op_type == "read_file":
        p = Path(op.get("path", ""))
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="replace")
            head = op.get("head")
            tail = op.get("tail")
            if head:
                content = "\\n".join(content.splitlines()[:head])
            elif tail:
                content = "\\n".join(content.splitlines()[-tail:])
            print(content)
        else:
            print("File not found:", str(p))

    elif op_type == "write_file":
        p = Path(op.get("path", ""))
        p.parent.mkdir(parents=True, exist_ok=True)
        content = op.get("content", "")
        p.write_text(content, encoding="utf-8")
        print("Written", len(content), "chars to", str(p))

    elif op_type == "append_file":
        p = Path(op.get("path", ""))
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(op.get("content", ""))
        print("Appended to", str(p))

    elif op_type == "edit_file":
        p = Path(op.get("path", ""))
        text = p.read_text(encoding="utf-8")
        old_text = op.get("old_text", "")
        new_text = op.get("new_text", "")
        if old_text in text:
            text = text.replace(old_text, new_text, 1)
            if not op.get("dry_run", False):
                p.write_text(text, encoding="utf-8")
                print("Edited", str(p))
            else:
                print("[DRY RUN] Would edit", str(p))
        else:
            print("old_text not found in", str(p))

    elif op_type == "list_directory":
        p = Path(op.get("path", "."))
        if p.exists():
            for e in sorted(p.iterdir()):
                kind = "DIR " if e.is_dir() else "FILE"
                print(" ", kind, e.name)
        else:
            print("Directory not found:", str(p))

    elif op_type == "directory_tree":
        p = Path(op.get("path", "."))
        max_depth = op.get("depth", 3)
        def _tree(tp, prefix="", d=0):
            if d >= max_depth:
                return
            try:
                entries = sorted(Path(tp).iterdir())
            except Exception:
                return
            for idx, e in enumerate(entries):
                connector = "└── " if idx == len(entries) - 1 else "├── "
                print(prefix + connector + e.name)
                if e.is_dir():
                    ext = "    " if idx == len(entries) - 1 else "│   "
                    _tree(e, prefix + ext, d + 1)
        print(str(p))
        _tree(p)

    elif op_type == "create_directory":
        p = Path(op.get("path", ""))
        p.mkdir(parents=True, exist_ok=True)
        print("Created directory:", str(p))

    elif op_type == "move_file":
        src, dst = op.get("src", ""), op.get("dst", "")
        shutil.move(src, dst)
        print("Moved", src, "->", dst)

    elif op_type == "copy_file":
        src, dst = op.get("src", ""), op.get("dst", "")
        if Path(src).is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print("Copied", src, "->", dst)

    elif op_type == "delete_file":
        p = Path(op.get("path", ""))
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
        print("Deleted:", str(p))

    elif op_type == "file_info":
        p = Path(op.get("path", ""))
        if p.exists():
            s = p.stat()
            print(json.dumps({{"path": str(p), "size": s.st_size, "is_dir": p.is_dir()}}))
        else:
            print("Not found:", str(p))

    elif op_type == "file_exists":
        p = Path(op.get("path", ""))
        print(str(p), "exists:", p.exists())

    elif op_type == "search_files":
        p = Path(op.get("path", "."))
        pattern = op.get("pattern", "*")
        for m in sorted(p.glob(pattern)):
            print(str(m))

    elif op_type in ("ensure_uv_available", "is_uv_environment"):
        r = subprocess.run(["which", "uv"], capture_output=True, text=True)
        print("uv available:", r.returncode == 0)

    elif op_type == "check_nodejs_availability":
        r = subprocess.run(["which", "node"], capture_output=True, text=True)
        print("nodejs available:", r.returncode == 0)

    else:
        print("[WARN] Unsupported op type:", op_type)


for _i, _op in enumerate(_OPS):
    _run_op(_op)
'''
    return code


class KnowledgeExecutor(BaseTrackExecutor):

    async def execute(
        self,
        skill: Skill,
        query: str,
        params: dict[str, Any],
    ) -> tuple[SkillExecutionResult, str]:
        effective_query = params.get("request") or query
        remaining_params = {k: v for k, v in params.items() if k != "request"}

        code_param = (remaining_params.get("code") or "").strip()
        if code_param:
            return await self._execute_code(skill, effective_query, code_param, remaining_params)

        prebuilt_ops = extract_commands_from_skill_md(skill.code, skill.source_dir)
        if prebuilt_ops:
            logger.info(
                "Knowledge skill '%s': pre-parsed %d command(s) from SKILL.md, "
                "bypassing LLM interpretation",
                skill.name,
                len(prebuilt_ops),
            )
            code = _ops_to_python(prebuilt_ops, skill.name)
            return await self._execute_code(skill, effective_query, code, remaining_params)

        return await self._execute_direct_llm(skill, effective_query, remaining_params)

    async def _execute_direct_llm(
        self,
        skill: Skill,
        query: str,
        params: dict[str, Any],
    ) -> tuple[SkillExecutionResult, str]:
        logger.info("Executing knowledge skill '%s' (direct LLM path)", skill.name)
        prompt = KNOWLEDGE_EXECUTE_PROMPT.format(
            skill_name=skill.name,
            description=skill.description or "(no description)",
            skill_content=skill.code,
            query=query,
            params_json=json.dumps(params, ensure_ascii=False, indent=2) if params else "{}",
            workspace_path=str(g_settings.workspace_path),
        )
        try:
            raw = await self._chat_async(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8192,
                timeout=180,
            )
            if not raw or not raw.strip():
                return SkillExecutionResult(
                    success=False, result=None,
                    error="LLM returned empty response for knowledge skill",
                    skill_name=skill.name,
                ), ""

            parsed = parse_knowledge_response(raw)
            if parsed is not None:
                if not parsed.get("relevant", True):
                    reason = parsed.get("reason", "skill not relevant")
                    logger.warning("Knowledge skill '%s' judged NOT relevant: %s", skill.name, reason)
                    return SkillExecutionResult(
                        success=False, result=reason,
                        error=f"Skill mismatch: '{skill.name}' — {reason}",
                        skill_name=skill.name,
                    ), ""
                content = parsed.get("content", raw)
            else:
                content = raw
                if detect_skill_mismatch(raw, skill.name, query):
                    return SkillExecutionResult(
                        success=False, result=raw,
                        error=f"Skill mismatch: '{skill.name}' is not relevant to query",
                        skill_name=skill.name,
                    ), ""

            ops = _extract_ops_json(content)
            if ops:
                logger.info(
                    "Knowledge skill '%s' returned %d ops, auto-executing...",
                    skill.name, len(ops),
                )
                code = _ops_to_python(ops, skill.name)
                return await self._execute_code(skill, query, code, params)

            python_code = _extract_executable_python(content)
            if python_code:
                logger.info(
                    "Knowledge skill '%s' returned Python code (%d bytes), auto-executing...",
                    skill.name, len(python_code),
                )
                return await self._execute_code(skill, query, python_code, params)

            logger.info("Knowledge skill '%s' completed (%d chars)", skill.name, len(content))
            return SkillExecutionResult(success=True, result=content, skill_name=skill.name), ""
        except Exception as e:
            logger.error("Knowledge skill '%s' failed: %s", skill.name, e)
            return SkillExecutionResult(
                success=False, result=None,
                error=f"{type(e).__name__}: {e}", skill_name=skill.name,
            ), ""

    async def _execute_code(
        self,
        skill: Skill,
        query: str,
        code: str,
        params: dict[str, Any],
    ) -> tuple[SkillExecutionResult, str]:
        logger.info("Knowledge skill '%s' executing code in workspace (%d bytes)", skill.name, len(code))
        result = await self._run_in_workspace(skill, code)
        if result.success:
            logger.info("Knowledge skill '%s' code succeeded (%d chars)", skill.name, len(result.result or ""))
            return result, code
        logger.warning("Knowledge skill '%s' code failed: %s", skill.name, (result.error or "")[:4000])
        if self.max_retries > 0:
            return await self._reflect_and_retry(skill, query, code, result.error or "Unknown error", params)
        return result, code

    async def _run_in_workspace(self, skill: Skill, code: str) -> SkillExecutionResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_in_workspace_sync, skill, code)

    def _run_in_workspace_sync(self, skill: Skill, code: str) -> SkillExecutionResult:
        workspace = g_settings.workspace_path
        workspace.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="delta_knowledge_") as tmp:
            runner_path = Path(tmp) / "__runner__.py"
            runner_path.write_text(code, encoding="utf-8")

            return self.sandbox.run(
                [sys.executable, str(runner_path)],
                cwd=workspace,
                pythonpath=workspace,
                timeout=g_settings.execution_timeout_sec,
                skill_name=skill.name,
                check_syntax=code,
            )
