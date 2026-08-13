# SPDX-License-Identifier: Apache-2.0
"""Skill routing and evolution maintenance commands."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from core.skill.evolution import SkillEvolutionEngine
from shared.schema import SkillConfig

skills_app = typer.Typer(no_args_is_help=True)
console = Console()


@skills_app.command("rollback")
def rollback(
    skill_name: str = typer.Argument(..., help="Installed skill name"),
) -> None:
    """Restore the latest pre-evolution snapshot for a skill."""

    async def _run() -> None:
        engine = SkillEvolutionEngine(
            config=SkillConfig.from_global_config(),
            llm=None,
            candidate_runner=None,
        )
        result = await engine.rollback_last(skill_name)
        if result.status != "rolled_back":
            console.print(f"[red]Rollback failed:[/red] {result.error}")
            raise typer.Exit(code=1)
        console.print(
            f"[green]Rolled back[/green] {result.skill_name} "
            f"from snapshot {result.event_id}"
        )

    asyncio.run(_run())
