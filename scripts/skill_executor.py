"""Standalone SkillExecutor test runner.

Usage examples:
  python scripts/skill_executor.py
  python scripts/skill_executor.py --list
  python scripts/skill_executor.py --skill filesystem --request "list files in ."

This script executes the full SkillGateway flow (provider -> executor) so you can
compare tool_calls / python fallback / text-only responses across skills.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from middleware.config import g_config
from core.skill.gateway import SkillGateway


def _print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


CASE_LIBRARY: list[dict[str, Any]] = [
    {
        "name": "filesystem_list",
        "skill": "filesystem",
        "request": "List files in the workspace root and include file sizes.",
    },
    {
        "name": "filesystem_read",
        "skill": "filesystem",
        "request": "Read the README.md and summarize its purpose in 3 bullets.",
    },
    {
        "name": "filesystem_missing_request",
        "skill": "filesystem",
        "request": "",
    },
    {
        "name": "web_search",
        "skill": "web-search",
        "request": "最新的 Python 3.13 版本有哪些新特性？",
    },
    {
        "name": "pdf_summary",
        "skill": "pdf",
        "request": "Summarize the first two pages of the pdf at /path/to/sample.pdf.",
    },
    {
        "name": "xlsx_overview",
        "skill": "xlsx",
        "request": "Open the spreadsheet at /path/to/sample.xlsx and list sheet names.",
    },
    {
        "name": "docx_extract",
        "skill": "docx",
        "request": "Extract headings from /path/to/sample.docx.",
    },
    {
        "name": "pptx_outline",
        "skill": "pptx",
        "request": "Generate a slide outline for /path/to/sample.pptx.",
    },
    {
        "name": "image_analysis",
        "skill": "image-analysis",
        "request": "Describe the main objects in /path/to/sample.png.",
    },
    {
        "name": "invalid_skill",
        "skill": "nonexistent_skill",
        "request": "This should fail with skill not found.",
    },
]


async def _list_skills(provider: SkillGateway) -> None:
    skills = await provider.discover()
    print(f"Loaded {len(skills)} skill(s) from {g_config.get_skills_path()}")
    for m in skills:
        print(f"- {m.name}: {m.description}")


async def _execute_skill(provider: SkillGateway, skill_name: str, request: str) -> None:
    response = await provider.execute(
        skill_name=skill_name,
        params={"request": request},
    )
    payload = {
        "ok": response.ok,
        "status": response.status.value,
        "error_code": response.error_code.value if response.error_code else None,
        "summary": response.summary,
        "skill_name": response.skill_name,
        "output": response.output,
        "outputs": response.outputs,
        "artifacts": response.artifacts,
        "diagnostics": response.diagnostics,
    }
    _print_payload(payload)


async def _prepare_dependency_case(provider: SkillGateway) -> dict[str, str] | None:
    workspace = Path(g_config.paths.workspace_dir)
    test_dir = workspace / "skill_executor" / "deps"
    test_dir.mkdir(parents=True, exist_ok=True)

    sample_path = test_dir / "sample.xlsx"
    if not sample_path.exists():
        import zipfile

        content_types = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">
  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>
  <Default Extension=\"xml\" ContentType=\"application/xml\"/>
  <Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>
  <Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>
</Types>
"""
        rels = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>
</Relationships>
"""
        workbook = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">
  <sheets>
    <sheet name=\"Sheet1\" sheetId=\"1\" r:id=\"rId1\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"/>
  </sheets>
</workbook>
"""
        workbook_rels = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>
</Relationships>
"""
        sheet1 = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">
  <sheetData>
    <row r=\"1\">
      <c r=\"A1\" t=\"str\"><v>hello</v></c>
    </row>
  </sheetData>
</worksheet>
"""
        with zipfile.ZipFile(sample_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", rels)
            zf.writestr("xl/workbook.xml", workbook)
            zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
            zf.writestr("xl/worksheets/sheet1.xml", sheet1)

    return {
        "name": "xlsx_dependency_auto_install",
        "skill": "xlsx",
        "request": f"Open {sample_path} and list sheet names.",
    }


async def _run_cases(provider: SkillGateway) -> None:
    dependency_case = await _prepare_dependency_case(provider)

    for case in CASE_LIBRARY:
        name = case.get("name", "unnamed")
        print(f"\n=== CASE: {name} ===")
        await _execute_skill(provider, case["skill"], case["request"])

    if dependency_case:
        print(f"\n=== CASE: {dependency_case['name']} ===")
        await _execute_skill(
            provider,
            dependency_case["skill"],
            dependency_case["request"],
        )


async def main() -> None:
    g_config.load()

    parser = argparse.ArgumentParser(description="Run SkillExecutor tests")
    parser.add_argument("--list", action="store_true", help="List local skills")
    parser.add_argument("--skill", type=str, help="Skill name to execute")
    parser.add_argument("--request", type=str, help="Request text for the skill")
    args = parser.parse_args()

    from core.skill.config import SkillConfig
    from core.skill.store import SkillStore

    config = SkillConfig.from_global_config()
    provider = await SkillGateway.from_config(config)

    if args.list:
        await _list_skills(provider)
        return

    if args.skill and args.request is not None:
        await _execute_skill(provider, args.skill, args.request)
        return

    await _run_cases(provider)


if __name__ == "__main__":
    asyncio.run(main())
