from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, AsyncGenerator
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from core.config import AGENT_SYSTEM_PROMPT_TEMPLATE, WORKSPACE_DIR
from core.memento_client import MementoToolManager

logger = logging.getLogger(__name__)

def _to_lc_messages(
    messages: list[dict],
) -> list[HumanMessage | AIMessage]:
    out: list[HumanMessage | AIMessage] = []
    for m in messages:
        content = str(m.get("content", ""))
        if m.get("role") == "user":
            out.append(HumanMessage(content=content))
        else:
            out.append(AIMessage(content=content))
    return out


class MementoAgent:
    _SYSTEM_PROMPT_TEMPLATE = AGENT_SYSTEM_PROMPT_TEMPLATE
    def __init__(
        self,
        *,
        model: BaseChatModel,
        system_prompt: str | None = None,
        base_dir: Path | None = None,
        recursion_limit: int = 150,
    ) -> None:
        self.model = model
        if system_prompt:
            self._system_prompt = system_prompt
        else:
            workspace = str(base_dir) if base_dir else str(WORKSPACE_DIR)
            self._system_prompt = self._SYSTEM_PROMPT_TEMPLATE.format(workspace=workspace)
        self._base_dir = base_dir
        self._recursion_limit = recursion_limit
        self._tool_manager = MementoToolManager()
        self._agent_graph: Any = None

    async def start(self) -> None:
        await self._tool_manager.start(base_dir=self._base_dir)
        tools = self._tool_manager.get_langchain_tools()
        logger.info(f"MementoAgent: loaded tools: {[t.name for t in tools]}")

        self._agent_graph = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=self._system_prompt,
        )

    async def run(self, query: str | list[dict]) -> dict[str, Any]:
        if not self._agent_graph:
            raise RuntimeError("Agent not started. Call start() first.")

        if isinstance(query, str):
            messages: list[HumanMessage | AIMessage] = [HumanMessage(content=query)]
        else:
            messages = _to_lc_messages(query)

        result = await self._agent_graph.ainvoke(
            {"messages": messages},
            config={"recursion_limit": self._recursion_limit},
        )
        return result

    async def stream(
        self,
        query: str | list[dict],
    ) -> AsyncGenerator[dict[str, Any], None]:
        if not self._agent_graph:
            raise RuntimeError("Agent not started. Call start() first.")

        if isinstance(query, str):
            messages: list[HumanMessage | AIMessage] = [HumanMessage(content=query)]
        else:
            messages = _to_lc_messages(query)

        async for chunk in self._agent_graph.astream(
            {"messages": messages},
            stream_mode="updates",
        ):
            yield chunk

    async def close(self) -> None:
        await self._tool_manager.shutdown()
        self._agent_graph = None

    @property
    def tool_manager(self) -> MementoToolManager:
        return self._tool_manager

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tool_manager.get_langchain_tools()]