# SPDX-License-Identifier: Apache-2.0
"""Skill tool constants and schemas shared across all handlers."""

from __future__ import annotations

from typing import Any

TOOL_SEARCH_SKILL = "search_skill"
TOOL_EXECUTE_SKILL = "execute_skill"
TOOL_DOWNLOAD_SKILL = "download_skill"
TOOL_CREATE_SKILL = "create_skill"
TOOL_ASK_USER = "ask_user"
TOOL_RECALL_CONTEXT = "recall_context"


SKILL_SEARCH_EXECUTE_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_SEARCH_SKILL,
            "description": "Search for relevant skills by natural language query across BOTH local installed skills and the remote skill server. Use this first when you don't know which skill to use.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language intent to search skills for.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Max number of candidate skills to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_EXECUTE_SKILL,
            "description": "Execute a LOCAL installed skill. MUST NOT be used to execute remote or non-existent skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Exact logic name of the skill to execute (e.g., 'weather_fetcher').",
                    },
                    "request": {
                        "type": "string",
                        "description": "Natural language description of what you want the skill to do.",
                    },
                },
                "required": ["skill_name", "request"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_DOWNLOAD_SKILL,
            "description": "Download and install a remote skill from the skill server to the local environment. Use this ONLY AFTER search_skill has found a matching remote skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The exact name of the remote skill found via search_skill.",
                    }
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_CREATE_SKILL,
            "description": "Create a NEW skill from scratch ONLY when search_skill returns NO results from both local and remote. This writes the skill to the local file system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "Natural language description of what skill to create, including name, purpose, language, and functionality details.",
                    },
                },
                "required": ["request"],
            },
        },
    },
]


TOOL_ASK_USER_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_ASK_USER,
        "description": "Ask the user a question when you need information that only the user can provide. The execution will pause until the user responds.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
            },
            "required": ["question"],
        },
    },
}


TOOL_RECALL_CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_RECALL_CONTEXT,
        "description": (
            "Retrieve context from earlier in this session. "
            "Query by tool_call_id to get a persisted artifact's full content, "
            "or by keyword to search across session memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A tool_call_id or keyword to search for.",
                },
            },
            "required": ["query"],
        },
    },
}


AGENT_TOOL_SCHEMAS: list[dict[str, Any]] = SKILL_SEARCH_EXECUTE_SCHEMAS + [
    TOOL_ASK_USER_SCHEMA,
    TOOL_RECALL_CONTEXT_SCHEMA,
]
