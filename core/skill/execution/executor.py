"""统一 Skill 执行器 — LLM 读 SKILL.md → tool_calls (含 bash) / 生成代码"""

from __future__ import annotations

import json
import os
import platform
import shlex
import sys
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.debug_logger import (
    log_skill_exec,
    log_sandbox_exec,
    log_pip_install,
    log_python_exec,
    log_error_context,
)
from middleware.llm import LLMClient
from middleware.config import g_config

from core.skill.schema import ErrorType, Skill, SkillExecutionOutcome

logger = get_logger(__name__)


class SkillExecutor:
    def __init__(self, *, sandbox=None, llm: Any = None):
        from core.memento_s.policies import PolicyManager
        from core.skill.execution.sandbox import get_sandbox

        _llm = llm if llm is not None else LLMClient()
        self._llm = _llm
        self._sandbox = sandbox or get_sandbox()
        self._policy_manager = PolicyManager()

    def _get_python_executable(self) -> str:
        if self._sandbox is not None:
            try:
                return str(self._sandbox.python_executable)
            except Exception:
                pass
        return sys.executable

    # ------------------------------------------------------------------ #
    #  入口
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        skill: Skill,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[SkillExecutionOutcome, str]:
        from builtin.tools.registry import BUILTIN_TOOL_SCHEMAS

        # DEBUG: 记录 Skill 执行开始
        log_skill_exec(skill.name, query, phase="start")

        try:
            # 根据 allowed_tools 过滤可用工具（渐进式披露）
            filtered_tools = self._filter_tools_by_allowed_list(
                BUILTIN_TOOL_SCHEMAS, skill.allowed_tools
            )

            # 根据 query 相关性选择 references（渐进式披露）
            selected_references = self._select_relevant_references(
                skill.references, query
            )

            prompt = self._build_prompt(
                skill,
                query,
                selected_references,
                params=params,
            )

            response = await self._llm.async_chat(
                messages=[{"role": "user", "content": prompt}],
                tools=filtered_tools,
            )

            if response.has_tool_calls:
                tool_names = [tc.name for tc in response.tool_calls]
                logger.info(
                    "LLM returned tool calls: count={}, tools={}",
                    len(response.tool_calls),
                    tool_names,
                )
                return await self._execute_with_tool_calls(skill, response.tool_calls)

            content = response.text
            logger.info(
                "LLM returned no tool calls. finish_reason={}, content_preview={}",
                response.finish_reason,
                (content or "")[:200],
            )
            if content.startswith("[NOT_RELEVANT]"):
                return SkillExecutionOutcome(
                    success=False,
                    result="",
                    error=content[len("[NOT_RELEVANT]") :].strip(),
                    skill_name=skill.name,
                ), ""

            return await self._execute_fallback(skill, content)
        finally:
            # DEBUG: 记录 Skill 执行结束
            log_skill_exec(skill.name, query, phase="end")

    # ------------------------------------------------------------------ #
    #  Prompt 构建
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        skill: Skill,
        query: str,
        selected_references: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        from core.skill.execution.prompts import SKILL_EXECUTE_PROMPT
        from builtin.tools.registry import get_tools_summary

        skill_content = self._get_skill_content(skill)
        if skill.is_playbook and skill.source_dir:
            scripts_section = self._build_scripts_section(skill)
            if scripts_section:
                skill_content = skill_content + "\n\n" + scripts_section

        from datetime import datetime

        now = datetime.now()
        platform_info = f"{platform.system()} {platform.machine()}"
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        current_year = str(now.year)
        previous_year = str(now.year - 1)

        output_dir = str(g_config.paths.workspace_dir)

        # 构建 references section（渐进式披露）
        references_section = ""
        if selected_references:
            ref_lines = ["## References\n"]
            for ref_name, ref_content in selected_references.items():
                ref_lines.append(f"### {ref_name}")
                ref_lines.append(
                    ref_content[:2000] if len(ref_content) > 2000 else ref_content
                )
                ref_lines.append("")
            references_section = "\n".join(ref_lines)

        # Skill params (structured disclosure)
        params_section = ""
        if params:
            try:
                params_json = json.dumps(params, ensure_ascii=False, indent=2)
            except Exception:
                params_json = str(params)
            params_section = (
                "\n---\n\n"
                "## Skill Parameters (Structured)\n"
                f"```json\n{params_json}\n```\n"
            )

        return SKILL_EXECUTE_PROMPT.format(
            skill_name=skill.name,
            description=skill.description or "",
            skill_content=skill_content,
            tools_summary=get_tools_summary(),
            query=query,
            output_dir=output_dir,
            platform_info=platform_info,
            current_datetime=current_datetime,
            current_year=current_year,
            previous_year=previous_year,
            references_section=references_section + params_section,
        )

    def _build_scripts_section(self, skill: Skill) -> str:
        """列出 skill 目录下可用脚本（排除 SKILL.md），拼入 prompt。"""
        skill_dir = Path(skill.source_dir)
        if not skill_dir.exists():
            return ""

        from core.utils.platform import _detect_script_extensions

        SCRIPT_EXTENSIONS = _detect_script_extensions()

        scripts = sorted(
            p
            for p in skill_dir.rglob("*")
            if p.is_file()
            and p.suffix in SCRIPT_EXTENSIONS
            and p.name not in {"__init__.py", "SKILL.md"}
        )
        if not scripts:
            return ""

        lines = ["## Available Scripts (absolute paths)"]
        for s in scripts:
            lines.append(f"- `{s}`")
        lines.append("")

        example_script = str(scripts[0])
        lines.append("Use `bash` tool to run these scripts, e.g.:")
        lines.append(f"  bash: python {example_script} <args>")
        lines.append(
            "IMPORTANT: Use the absolute paths above directly. Do NOT cd to workspace before running scripts."
        )
        lines.append("Note: `python` will be resolved to the uv sandbox interpreter.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  tool_calls 执行
    # ------------------------------------------------------------------ #

    async def _execute_with_tool_calls(
        self,
        skill: Skill,
        tool_calls: list,
    ) -> tuple[SkillExecutionOutcome, str]:
        from builtin.tools import execute_builtin_tool
        from builtin.tools.registry import is_builtin_tool, get_tool_schema

        results = []
        blocked_reason = None
        blocked_error_type: ErrorType | None = None
        first_error_type: ErrorType | None = None
        first_error_detail: dict[str, Any] | None = None

        for idx, tc in enumerate(tool_calls, start=1):
            tool_name = (
                tc.name
                if hasattr(tc, "name")
                else tc.get("function", {}).get("name", "")
            )
            if hasattr(tc, "arguments"):
                raw_args = tc.arguments
            else:
                arguments_str = tc.get("function", {}).get("arguments", "{}")
                raw_args = (
                    json.loads(arguments_str)
                    if isinstance(arguments_str, str)
                    else arguments_str
                )

            if not tool_name:
                results.append({"skipped": True, "reason": "no_tool_name"})
                continue

            if not is_builtin_tool(tool_name):
                results.append(
                    {
                        "index": idx,
                        "tool": tool_name,
                        "skipped": True,
                        "reason": "not_builtin_tool",
                    }
                )
                continue

            args = self._normalize_args(tool_name, raw_args)
            if skill.source_dir:
                schema = get_tool_schema(tool_name) or {}
                props = schema.get("properties", {})

                session_id = "default"
                workspace_dir = str(g_config.paths.workspace_dir)

                # Resolve relative paths against skill directory when they exist there
                args = self._maybe_resolve_tool_paths(args, skill, tool_name, props)

                # Path resolution base_dir:
                # - For bash: base_dir can point to the resolved skill root for referencing skill-local resources.
                # - For file tools: base_dir should be the workspace_dir so relative paths like '.' and '..'
                #   resolve within the workspace (and within data_dir boundary).
                skill_root = Path(skill.source_dir) if skill.source_dir else None
                try:
                    skills_dir = Path(g_config.paths.skills_dir)
                    candidate = skills_dir / skill.name
                    if candidate.exists():
                        skill_root = candidate
                except Exception:
                    pass

                if "base_dir" in props and "base_dir" not in args:
                    if tool_name == "bash" and skill_root:
                        args["base_dir"] = str(skill_root)
                    else:
                        args["base_dir"] = workspace_dir

                # Execution cwd for bash
                if "work_dir" in props and "work_dir" not in args:
                    args["work_dir"] = workspace_dir

            if tool_name == "bash":
                args = self._maybe_rewrite_bash_input(args)
                policy = self._policy_manager.check("bash", args)
                if not policy.allowed:
                    reason = policy.reason or "bash command denied by policy"
                    logger.warning(
                        "Tool call blocked by policy: {} -> {}", tool_name, reason
                    )
                    blocked_reason = reason
                    blocked_error_type = ErrorType.POLICY_BLOCKED
                    break

            logger.info("!Tool call start: #{} tool={} args={}", idx, tool_name, args)

            # DEBUG: 记录沙箱执行
            if tool_name == "bash":
                cmd = args.get("command", "")
                log_sandbox_exec(cmd, args.get("work_dir"), {"tool": tool_name})

            try:
                tool_result = await execute_builtin_tool(tool_name, args)
                # 对于 search_web，在中间日志中不打印完整结果
                if tool_name == "search_web":
                    # 提取搜索结果的概要信息
                    if isinstance(tool_result, str):
                        # 如果是字符串，只显示前 200 字符
                        summary = (
                            tool_result[:200] + "..."
                            if len(tool_result) > 200
                            else tool_result
                        )
                    elif isinstance(tool_result, dict) and "results" in tool_result:
                        # 如果是字典，显示结果数量
                        results_count = len(tool_result.get("results", []))
                        summary = f"[搜索完成，返回 {results_count} 条结果]"
                    else:
                        summary = "[搜索完成]"
                    logger.info(
                        f"Tool call done: #{idx} tool={tool_name} result={summary}"
                    )
                else:
                    result_json = json.dumps(
                        tool_result, ensure_ascii=False, default=str
                    )
                    logger.info(
                        f"Tool call done: #{idx} tool={tool_name} result={result_json}"
                    )

                warning = None
                if tool_name == "bash" and isinstance(tool_result, str):
                    if self._is_nonfatal_http_error(tool_result):
                        retry_result = await self._maybe_retry_bash_on_http_error(
                            args, tool_result
                        )
                        if retry_result is not None:
                            tool_result = retry_result
                        else:
                            warning = "Non-fatal HTTP error detected (request failure)."

                entry = {
                    "index": idx,
                    "tool": tool_name,
                    "args": args,
                    "result": tool_result,
                }
                if warning:
                    entry["warning"] = warning
                results.append(entry)

                if isinstance(tool_result, str) and not warning:
                    classified = self._classify_tool_error(tool_name, tool_result)
                    if classified:
                        if first_error_type is None:
                            first_error_type = classified[0]
                            first_error_detail = classified[1]
            except Exception as e:
                logger.warning(
                    "Tool call failed: #{} tool={} err={}", idx, tool_name, e
                )
                results.append(
                    {"index": idx, "tool": tool_name, "args": args, "error": str(e)}
                )
                if first_error_type is None:
                    first_error_type = ErrorType.INTERNAL_ERROR
                    first_error_detail = {"message": str(e)}

        output = f"[Executed {len(tool_calls)} tool call(s)]\n"
        output += json.dumps(results, ensure_ascii=False, indent=2)

        return SkillExecutionOutcome(
            success=blocked_reason is None and first_error_type is None,
            result=output,
            error=(
                f"Blocked by policy: {blocked_reason}"
                if blocked_reason
                else (first_error_detail.get("message") if first_error_detail else None)
            ),
            error_type=blocked_error_type or first_error_type,
            error_detail=first_error_detail,
            skill_name=skill.name,
            operation_results=results,
        ), ""

    # ------------------------------------------------------------------ #
    #  Fallback：无 tool_calls 时处理 LLM 返回内容
    # ------------------------------------------------------------------ #

    async def _execute_fallback(
        self,
        skill: Skill,
        llm_content: str,
    ) -> tuple[SkillExecutionOutcome, str]:
        from builtin.tools import execute_builtin_tool

        code = self._extract_code_block(llm_content)

        # 无代码块 → 纯文本回答（知识/指导），直接返回
        if code is None:
            return SkillExecutionOutcome(
                success=True,
                result=llm_content,
                skill_name=skill.name,
            ), ""

        # 有代码块 → 执行（强制走 sandbox）
        policy = self._policy_manager.check("bash", {"command": "python -c ..."})
        if not policy.allowed:
            return SkillExecutionOutcome(
                success=False,
                result="",
                error=f"Blocked by policy: {policy.reason}",
                skill_name=skill.name,
            ), code

        try:
            result = self._sandbox.run_code(code, skill)
            return SkillExecutionOutcome(
                success=result.success,
                result=result.result,
                error=result.error,
                error_type=result.error_type,
                error_detail=result.error_detail,
                skill_name=skill.name,
                artifacts=result.artifacts,
                operation_results=result.operation_results,
            ), code
        except Exception as e:
            return SkillExecutionOutcome(
                success=False,
                result="",
                error=f"Execution failed: {e}",
                skill_name=skill.name,
            ), code

    @staticmethod
    def _extract_code_block(text: str) -> str | None:
        """从 markdown 中提取代码块。有代码块返回代码，没有返回 None。"""
        if "```" not in text:
            return None

        lines = text.split("\n")
        in_block = False
        code_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                code_lines.append(line)

        if not code_lines:
            return None
        return "\n".join(code_lines)

    @staticmethod
    def _maybe_rewrite_bash_input(args: dict[str, Any]) -> dict[str, Any]:
        """Rewrite bash command to use stdin when it embeds a large quoted payload.

        This avoids shell-quoting failures when JSON/code is embedded in single quotes.
        If stdin is already provided, keep it unchanged.
        """
        if not isinstance(args, dict):
            return args
        if args.get("stdin") is not None:
            return args

        command = args.get("command")
        if not isinstance(command, str) or "--input" not in command:
            return args

        def _extract_quoted_payload(cmd: str) -> tuple[str, str] | None:
            for quote in ("'", '"'):
                token = f"--input {quote}"
                idx = cmd.find(token)
                if idx == -1:
                    continue
                start = idx + len(token)
                end = cmd.find(quote, start)
                if end == -1:
                    continue
                payload = cmd[start:end]
                new_cmd = cmd[:idx] + "--input -" + cmd[end + 1 :]
                return new_cmd, payload
            return None

        extracted = _extract_quoted_payload(command)
        if not extracted:
            return args

        new_cmd, payload = extracted
        new_args = dict(args)
        new_args["command"] = new_cmd
        new_args["stdin"] = payload
        return new_args

    @staticmethod
    def _maybe_resolve_tool_paths(
        args: dict[str, Any],
        skill: Skill,
        tool_name: str,
        props: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(args, dict):
            return args
        if not skill.source_dir:
            return args

        try:
            max_name_len = os.pathconf(skill.source_dir, "PC_NAME_MAX")
        except (AttributeError, ValueError, OSError):
            max_name_len = 255

        def _is_relative_path(value: str) -> bool:
            if not value or not isinstance(value, str):
                return False
            if "\n" in value or "\r" in value or "\t" in value:
                return False
            if len(value) > max_name_len:
                return False
            if Path(value).is_absolute() or value.startswith("~"):
                return False
            return True

        def _resolve_candidate(value: str) -> str | None:
            try:
                candidate = Path(skill.source_dir) / value
                return str(candidate) if candidate.exists() else None
            except OSError:
                return None

        new_args = dict(args)

        # Special handling for bash command: resolve relative executables/scripts
        # Handles both simple ("python script.py") and compound ("cd /x && python script.py") commands.
        if tool_name == "bash":
            command = new_args.get("command")
            if isinstance(command, str) and command.strip():
                import re

                segments = re.split(r"(&&|;|\|\|)", command)
                changed = False

                for i, seg in enumerate(segments):
                    stripped = seg.strip()
                    if not stripped or stripped in {"&&", ";", "||"}:
                        continue

                    try:
                        parts = shlex.split(stripped)
                    except ValueError:
                        continue

                    if not parts:
                        continue

                    # Case 1: ./scripts/xxx or scripts/xxx executable
                    first = parts[0]
                    if _is_relative_path(first):
                        resolved = _resolve_candidate(first)
                        if resolved:
                            parts[0] = resolved
                            segments[i] = seg[
                                : len(seg) - len(seg.lstrip())
                            ] + " ".join(shlex.quote(p) for p in parts)
                            changed = True
                            continue

                    # Case 2: python <relative_script>
                    if len(parts) >= 2 and parts[0] in {"python", "python3"}:
                        script_arg = parts[1]
                        if _is_relative_path(script_arg):
                            resolved = _resolve_candidate(script_arg)
                            if resolved:
                                parts[1] = resolved
                                # If script is .sh, run via bash instead of python
                                if str(resolved).endswith(".sh"):
                                    parts[0] = "bash"
                                segments[i] = seg[
                                    : len(seg) - len(seg.lstrip())
                                ] + " ".join(shlex.quote(p) for p in parts)
                                changed = True

                if changed:
                    new_args["command"] = "".join(segments)
            return new_args

        _SKIP_KEYS = {
            "base_dir",
            "work_dir",
            "content",
            "stdin",
            "text",
            "data",
            "body",
        }
        # For file tools: resolve any path-like string args to skill dir if exists
        for key, value in args.items():
            if key in _SKIP_KEYS:
                continue
            if isinstance(value, str) and _is_relative_path(value):
                resolved = _resolve_candidate(value)
                if resolved:
                    new_args[key] = resolved

        return new_args

    # ------------------------------------------------------------------ #
    #  工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_skill_content(skill: Skill) -> str:
        """优先读 SKILL.md，fallback 到 skill.content。"""
        if skill.source_dir:
            skill_md = Path(skill.source_dir) / "SKILL.md"
            if skill_md.exists():
                return skill_md.read_text("utf-8")
        return skill.content

    def _normalize_args(self, tool_name: str, raw_args: dict) -> dict:
        """按 schema 规范化参数：过滤未知字段、required 校验、default 填充、类型转换。"""
        from builtin.tools.registry import get_tool_schema

        schema = get_tool_schema(tool_name)
        if not schema:
            return raw_args

        props = schema.get("properties", {})
        required = schema.get("required", [])

        for req_param in required:
            if req_param not in raw_args:
                logger.warning(
                    "Missing required parameter '{}' for tool '{}'",
                    req_param,
                    tool_name,
                )

        normalized = {}

        for param_name, param_info in props.items():
            if param_name in raw_args:
                value = raw_args[param_name]
                param_type = param_info.get("type")
                if param_type == "integer" and value is not None:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Cannot convert '{}' to integer for '{}'", value, param_name
                        )
                        default = param_info.get("default")
                        if default is not None:
                            value = default
                        else:
                            continue
                elif param_type == "boolean":
                    value = bool(value) if not isinstance(value, bool) else value
                normalized[param_name] = value
            elif param_name in required:
                pass
            else:
                default = param_info.get("default")
                if default is not None:
                    normalized[param_name] = default

        return normalized

    @staticmethod
    def _is_nonfatal_http_error(tool_output: str) -> bool:
        if not isinstance(tool_output, str):
            return False
        lower = tool_output.lower()
        return "http error" in lower and "client error" in lower

    async def _maybe_retry_bash_on_http_error(
        self,
        args: dict[str, Any],
        tool_output: str,
    ) -> str | None:
        from builtin.tools import execute_builtin_tool

        if not isinstance(tool_output, str):
            return None
        lower = tool_output.lower()
        if "client error" not in lower:
            return None

        command = str(args.get("command", ""))
        if not command:
            return None

        if "http" not in command:
            return None

        # Avoid infinite loops if already retried
        if "__http_retry__" in command:
            return None

        retry_args = dict(args)
        retry_args["command"] = f"__http_retry__=1 {command}"
        logger.info("Retrying bash command after HTTP client error")
        return await execute_builtin_tool("bash", retry_args)

    @staticmethod
    def _classify_tool_error(
        tool_name: str, tool_output: str
    ) -> tuple[ErrorType, dict[str, Any]] | None:
        """基于 tool 输出字符串，识别通用错误类型。"""
        text = tool_output.strip()
        lower = text.lower()

        if lower.startswith("err:"):
            if "timed out" in lower or "timeout" in lower:
                return ErrorType.TIMEOUT, {"tool": tool_name, "message": text}
            if "permission denied" in lower or "permission" in lower:
                return ErrorType.PERMISSION_DENIED, {"tool": tool_name, "message": text}
            return ErrorType.EXECUTION_ERROR, {"tool": tool_name, "message": text}

        if lower.startswith("exit code:"):
            if "command not found" in lower or "not found" in lower:
                return ErrorType.RESOURCE_MISSING, {"tool": tool_name, "message": text}
            if "permission denied" in lower:
                return ErrorType.PERMISSION_DENIED, {"tool": tool_name, "message": text}
            if "no such file or directory" in lower:
                return ErrorType.RESOURCE_MISSING, {"tool": tool_name, "message": text}
            if (
                "no such host" in lower
                or "temporary failure in name resolution" in lower
            ):
                return ErrorType.UNAVAILABLE, {"tool": tool_name, "message": text}
            return ErrorType.EXECUTION_ERROR, {"tool": tool_name, "message": text}

        if "missing required" in lower or "required parameter" in lower:
            return ErrorType.INPUT_REQUIRED, {"tool": tool_name, "message": text}

        if "invalid" in lower and "input" in lower:
            return ErrorType.INPUT_INVALID, {"tool": tool_name, "message": text}

        return None

    # ------------------------------------------------------------------ #
    #  渐进式披露工具方法
    # ------------------------------------------------------------------ #

    def _filter_tools_by_allowed_list(
        self, all_tools: list[dict[str, Any]], allowed_tools: list[str]
    ) -> list[dict[str, Any]]:
        """根据 skill 的 allowed_tools 过滤可用工具。

        Args:
            all_tools: 所有可用工具的 schema 列表
            allowed_tools: skill 声明的允许工具列表

        Returns:
            过滤后的工具 schema 列表。如果 allowed_tools 为空，返回全部工具。
        """
        if not allowed_tools:
            return all_tools

        allowed_set = set(allowed_tools)
        filtered = []
        for tool in all_tools:
            tool_name = tool.get("function", {}).get("name", "")
            if tool_name in allowed_set:
                filtered.append(tool)

        if not filtered:
            logger.warning(
                "Skill allowed_tools {} matched no tools, falling back to all tools",
                allowed_tools,
            )
            return all_tools

        logger.debug(
            "Filtered tools from {} to {} based on allowed_tools: {}",
            len(all_tools),
            len(filtered),
            allowed_tools,
        )
        return filtered

    def _select_relevant_references(
        self, references: dict[str, str], query: str, max_chars: int = 4000
    ) -> dict[str, str]:
        """根据 query 相关性选择 references（渐进式披露）。

        简单实现：优先选择文件名或内容与 query 关键词匹配的 references。
        生产环境可以使用 embedding 相似度计算。

        Args:
            references: 所有可用的 references {filename: content}
            query: 用户请求
            max_chars: 最大总字符数限制

        Returns:
            选中的 references 子集
        """
        if not references:
            return {}

        # 简单的关键词匹配
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_refs = []
        for ref_name, ref_content in references.items():
            score = 0
            ref_name_lower = ref_name.lower()
            ref_content_lower = ref_content.lower()

            # 文件名匹配加分
            if any(word in ref_name_lower for word in query_words if len(word) > 3):
                score += 2

            # 内容匹配加分
            content_matches = sum(
                1 for word in query_words if len(word) > 3 and word in ref_content_lower
            )
            score += content_matches * 0.5

            scored_refs.append((ref_name, ref_content, score))

        # 按分数降序排序
        scored_refs.sort(key=lambda x: x[2], reverse=True)

        # 选择 references，直到达到字符限制
        selected = {}
        total_chars = 0
        for ref_name, ref_content, score in scored_refs:
            if score > 0 or len(selected) == 0:  # 至少选一个，优先选相关的
                if total_chars + len(ref_content) > max_chars:
                    # 如果超出限制，截取部分内容
                    remaining = max_chars - total_chars
                    if remaining > 500:  # 至少要有意义的长度
                        selected[ref_name] = (
                            ref_content[:remaining] + "\n... (truncated)"
                        )
                        total_chars += remaining
                    break
                selected[ref_name] = ref_content
                total_chars += len(ref_content)

        logger.debug(
            "Selected {} references ({} chars) from {} total for query",
            len(selected),
            total_chars,
            len(references),
        )
        return selected
