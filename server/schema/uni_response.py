# SPDX-License-Identifier: Apache-2.0
"""
server/schema/uni_response.py
统一响应格式 (UniResponse)
参照 docs/uni_response_design.md 设计实现
"""
import json
from datetime import datetime
from typing import Any
from uuid import uuid4
import json

from pydantic import BaseModel, Field


class UniResponse(BaseModel):
    """统一响应格式（扁平结构）

    所有接口返回使用相同的响应结构，字段按需填充，非必需字段为 None。

    字段说明：
    - type: 消息类型，使用 AGUIEventType 的值
    - event_id: 统一事件 ID，根据 type 不同代表不同含义：
        - TOOL_CALL_* 类型时 = tool_call_id
        - TEXT_MESSAGE_* 类型时 = message_id
        - 其他事件时 = 事件唯一标识
    """

    # === 核心标识 ===
    type: str = Field(description="消息类型（AGUIEventType 值）")
    trace_id: str | None = Field(default=None, description="追踪 ID")

    # === 时间与会话 ===
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    session_id: str | None = Field(default=None, description="会话 ID")
    conversation_id: str | None = Field(default=None, description="本轮对话 ID，user 和 assistant 消息共享，通过 role 区分")
    reply_to: str | None = Field(default=None, description="关联的上一条消息 ID")

    # === 消息内容 ===
    role: str | None = Field(default=None, description="角色: user|assistant|system|tool")
    content: str | None = Field(default=None, description="文本内容")

    # === 事件标识（统一 event_id）===
    event_id: str | None = Field(
        default=None,
        description="统一事件 ID: TOOL_CALL_* 时为 tool_call_id, TEXT_MESSAGE_* 时为 message_id, 其他为事件唯一标识"
    )

    # === 事件相关 ===
    run_id: str | None = Field(default=None, description="Agent 运行 ID")
    thread_id: str | None = Field(default=None, description="线程 ID")
    step: int | None = Field(default=None, description="当前步骤")
    status: str | None = Field(default=None, description="状态")
    payload: dict | None = Field(default=None, description="事件负载: TEXT_MESSAGE_CONTENT 时 {content: ...}, 其余类型时 {messages: ...}")
    tool_name: str | None = Field(default=None, description="工具名称")
    arguments: dict | None = Field(default=None, description="工具参数")
    result: Any | None = Field(default=None, description="工具执行结果")

    # === 响应相关 ===
    output_text: str | None = Field(default=None, description="最终输出文本")
    reason: str | None = Field(default=None, description="结束原因")
    steps: int | None = Field(default=None, description="总步骤数")
    duration_seconds: float | None = Field(default=None, description="耗时")
    model_name: str | None = Field(default=None, description="使用的模型")

    # === Token 用量 ===
    usage: dict | None = Field(default=None, description="Token 用量信息")
    context_tokens: int | None = Field(default=None, description="上下文 Token 数")

    # === 错误相关 ===
    code: str | None = Field(default=None, description="错误码")
    error: str | None = Field(default=None, description="错误信息")
    detail: str | None = Field(default=None, description="错误详情")

    # === 元数据 ===
    meta: dict | None = Field(default=None, description="扩展元数据")

    # ==================== 工厂方法 ====================

    @classmethod
    def new_event_id(cls) -> str:
        """生成新的事件 ID"""
        return f"evt_{uuid4().hex[:12]}"

    @classmethod
    def new_message_id(cls) -> str:
        """生成新的消息 ID"""
        return f"msg_{uuid4().hex[:12]}"

    @classmethod
    def new_tool_call_id(cls) -> str:
        """生成新的工具调用 ID"""
        return f"call_{uuid4().hex[:12]}"

    @classmethod
    def run_started(
        cls,
        run_id: str,
        thread_id: str | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """RUN_STARTED 事件"""
        return cls(
            type="RUN_STARTED",
            event_id=cls.new_event_id(),
            run_id=run_id,
            thread_id=thread_id,
            trace_id=trace_id,
            payload={"messages": "", "content": ""},
        )

    @classmethod
    def step_started(
        cls,
        step: int,
        run_id: str,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """STEP_STARTED 事件"""
        return cls(
            type="STEP_STARTED",
            event_id=cls.new_event_id(),
            step=step,
            run_id=run_id,
            trace_id=trace_id,
            payload={"messages": "", "content": ""},
        )

    @classmethod
    def step_finished(
        cls,
        step: int,
        status: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """STEP_FINISHED 事件"""
        return cls(
            type="STEP_FINISHED",
            event_id=cls.new_event_id(),
            step=step,
            status=status,
            run_id=run_id,
            trace_id=trace_id,
            payload={"messages": status or "", "content": ""},
        )

    @classmethod
    def text_message_start(
        cls,
        run_id: str,
        message_id: str | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """TEXT_MESSAGE_START 事件"""
        return cls(
            type="TEXT_MESSAGE_START",
            event_id=message_id or cls.new_message_id(),
            run_id=run_id,
            trace_id=trace_id,
            payload={"messages": "", "content": ""},
        )

    @classmethod
    def text_content(
        cls,
        delta: str,
        message_id: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """TEXT_MESSAGE_CONTENT 事件（流式文本片段）"""
        return cls(
            type="TEXT_MESSAGE_CONTENT",
            event_id=message_id or cls.new_message_id(),
            payload={"messages": "", "content": delta},
            run_id=run_id,
            trace_id=trace_id,
        )

    @classmethod
    def text_message_end(
        cls,
        message_id: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """TEXT_MESSAGE_END 事件"""
        return cls(
            type="TEXT_MESSAGE_END",
            event_id=message_id or cls.new_message_id(),
            run_id=run_id,
            trace_id=trace_id,
            payload={"messages": "", "content": ""},
        )

    @classmethod
    def tool_call_start(
        cls,
        tool_name: str,
        arguments: dict | None = None,
        tool_call_id: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """TOOL_CALL_START 事件"""
        return cls(
            type="TOOL_CALL_START",
            event_id=tool_call_id or cls.new_tool_call_id(),
            tool_name=tool_name,
            arguments=arguments,
            run_id=run_id,
            trace_id=trace_id,
            payload={"messages": tool_name or "", "content": ""},
        )

    @classmethod
    def tool_call_result(
        cls,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """TOOL_CALL_RESULT 事件"""
        result_str = str(result) if result is not None else ""
        # 提取可读内容：优先 final_response，其次 summary，兜底空字符串
        _content = ""
        try:
            _obj = json.loads(result_str) if isinstance(result, str) else (result or {})
            _content = (
                (_obj.get("output") or {}).get("final_response", "")
                or _obj.get("summary", "")
            )
        except Exception:
            pass
        return cls(
            type="TOOL_CALL_RESULT",
            event_id=tool_call_id or cls.new_tool_call_id(),
            tool_name=tool_name,
            result=result,
            run_id=run_id,
            trace_id=trace_id,
            payload={"messages": "", "content": _content, "raw_result": result_str},
        )

    @classmethod
    def run_finished(
        cls,
        output_text: str,
        reason: str | None = None,
        usage: dict | None = None,
        context_tokens: int | None = None,
        steps: int | None = None,
        duration_seconds: float | None = None,
        model_name: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """RUN_FINISHED 事件"""
        return cls(
            type="RUN_FINISHED",
            event_id=cls.new_event_id(),
            output_text=output_text,
            reason=reason,
            usage=usage,
            context_tokens=context_tokens,
            steps=steps,
            duration_seconds=duration_seconds,
            model_name=model_name,
            run_id=run_id,
            trace_id=trace_id,
            payload={"messages": output_text or "", "content": ""},
        )

    @classmethod
    def run_error(
        cls,
        error: str,
        detail: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """RUN_ERROR 事件"""
        return cls(
            type="RUN_ERROR",
            event_id=cls.new_event_id(),
            error=error,
            detail=detail,
            run_id=run_id,
            trace_id=trace_id,
            payload={"messages": error or "", "content": ""},
        )

    @classmethod
    def error_response(
        cls,
        error: str,
        code: str | None = None,
        detail: str | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """通用 ERROR 响应"""
        return cls(
            type="ERROR",
            event_id=cls.new_event_id(),
            error=error,
            code=code,
            detail=detail,
            trace_id=trace_id,
            payload={"messages": error or "", "content": ""},
        )

    @classmethod
    def data(
        cls,
        meta: dict | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """通用 DATA 响应"""
        return cls(
            type="DATA",
            event_id=cls.new_event_id(),
            meta=meta,
            trace_id=trace_id,
            payload={"messages": "", "content": ""},
        )

    @classmethod
    def ack(
        cls,
        meta: dict | None = None,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """ACK 确认消息"""
        return cls(
            type="ACK",
            event_id=cls.new_event_id(),
            meta=meta,
            trace_id=trace_id,
            payload={"messages": "", "content": ""},
        )

    @classmethod
    def session_created(
        cls,
        session_id: str,
        trace_id: str | None = None,
    ) -> "UniResponse":
        """SESSION_CREATED 事件（内部扩展类型）"""
        return cls(
            type="SESSION_CREATED",
            event_id=cls.new_event_id(),
            session_id=session_id,
            trace_id=trace_id,
            payload={"messages": "", "content": ""},
        )

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "UniResponse":
        """从旧格式事件字典创建 UniResponse（兼容性转换）

        处理旧格式字段名映射：
        - runId -> run_id
        - threadId -> thread_id
        - sessionId -> session_id
        - message -> error (当 type 为 error/ERROR 时)
        - delta -> delta (保持不变)
        """
        if not isinstance(event, dict):
            return event

        # 复制一份，避免修改原字典
        data = dict(event)

        # 字段名映射：camelCase -> snake_case
        field_mapping = {
            "runId": "run_id",
            "threadId": "thread_id",
            "sessionId": "session_id",
            "messageId": "message_id",
            "toolCallId": "tool_call_id",
            "toolName": "tool_name",
            "outputText": "output_text",
            "inputText": "input_text",
            "contextTokens": "context_tokens",
            "modelName": "model_name",
            "durationSeconds": "duration_seconds",
        }
        for old_key, new_key in field_mapping.items():
            if old_key in data:
                data[new_key] = data.pop(old_key)

        # 兼容旧 error 格式: {"type": "error", "message": "..."}
        if data.get("type") in ("error", "ERROR"):
            if "message" in data and "error" not in data:
                data["error"] = data.pop("message")
            data["type"] = "ERROR"

        # 类型清理：移除或转换不匹配的字段类型，避免 Pydantic 验证错误
        # 注意：PLAN_GENERATED 的 steps 是 list，需提前保存
        plan_steps = data.get("steps") if isinstance(data.get("steps"), list) else None

        int_fields = ["step", "steps", "context_tokens"]
        for field in int_fields:
            if field in data and data[field] is not None:
                if not isinstance(data[field], int):
                    data[field] = None

        float_fields = ["duration_seconds"]
        for field in float_fields:
            if field in data and data[field] is not None:
                if not isinstance(data[field], (int, float)):
                    data[field] = None

        dict_fields = ["usage", "arguments", "meta"]
        for field in dict_fields:
            if field in data and data[field] is not None:
                if not isinstance(data[field], dict):
                    data[field] = None

        # 根据 type 填充 payload
        # TOOL_CALL_START / TOOL_CALL_RESULT 强制重建，避免上游混入脏数据
        _force_rebuild = data.get("type", "") in ("TOOL_CALL_START", "TOOL_CALL_RESULT")
        if "payload" not in data or data["payload"] is None or _force_rebuild:
            event_type = data.get("type", "")
            if event_type == "TEXT_MESSAGE_CONTENT":
                data["payload"] = {"messages": "", "content": data.get("delta", "")}
            else:
                _summary_field = {
                    "TOOL_CALL_START": "tool_name",
                    "TOOL_CALL_RESULT": "result",
                    "RUN_FINISHED": "output_text",
                    "RUN_ERROR": "error",
                    "STEP_FINISHED": "status",
                    "INTENT_RECOGNIZED": "task",
                    "PLAN_GENERATED": "goal",
                    "REFLECTION_RESULT": "decision",
                }.get(event_type, "")
                msg = data.get(_summary_field, "") or ""
                if event_type == "TOOL_CALL_RESULT":
                    # 尝试解析 result 字段，提取 output.final_response
                    result_raw = data.get("result", "")
                    try:
                        if isinstance(result_raw, str):
                            result_obj = json.loads(result_raw)
                        else:
                            result_obj = result_raw

                        final_response = result_obj.get("output", {}).get("final_response", "")
                        if final_response:
                            payload = {
                                "messages": "",
                                "content": final_response,
                                "raw_result": result_obj,
                            }
                        else:
                            summary = result_obj.get("summary", "")
                            payload = {
                                "messages": "",
                                "content": summary if summary else str(msg) if msg else "",
                                "raw_result": result_obj,
                            }
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        payload = {"messages": "", "content": str(msg) if msg else ""}
                else:
                    payload = {"messages": str(msg) if msg else "", "content": ""}
                if event_type == "PLAN_GENERATED":
                    payload["steps"] = plan_steps or []
                elif event_type == "TOOL_CALL_START":
                    raw_arguments = data.get("arguments") or {}
                    if isinstance(raw_arguments, dict):
                        payload["arguments"] = raw_arguments.get("request", raw_arguments)
                    else:
                        payload["arguments"] = raw_arguments
                elif event_type == "REFLECTION_RESULT":
                    payload["arguments"] = {
                        "decision": data.get("decision", ""),
                        "reason": data.get("reason", ""),
                    }
                data["payload"] = payload

        return cls(**data)

    # ==================== 格式化方法 ====================

    def to_sse(self) -> str:
        """转换为 SSE 格式字符串"""
        return f"data: {self.model_dump_json(exclude_none=True, warnings=False)}\n\n"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于 JSON 响应）"""
        return self.model_dump(mode='json')

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
