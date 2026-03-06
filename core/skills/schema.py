
from __future__ import annotations

from dataclasses import dataclass
from typing import Any



def skill_to_openai_tool(skill_name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": skill_name,
            "description": description,
            "parameters": parameters,
        },
    }



@dataclass
class SkillCall:
    id: str
    name: str
    arguments: dict[str, Any]



@dataclass
class SkillResult:
    skill_call_id: str
    name: str
    result: str
    error: bool = False
