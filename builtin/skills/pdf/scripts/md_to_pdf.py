#!/usr/bin/env python3
"""Convert a Markdown file to a styled PDF with CJK (Chinese) support.

Usage:
    python md_to_pdf.py <input.md> <output.pdf>

Dependencies: reportlab
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_CJK_FONT: str = "Helvetica"
_CJK_FONT_BOLD: str = "Helvetica-Bold"
_CJK_AVAILABLE: bool = False
try:
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    _CJK_FONT = "STSong-Light"
    _CJK_FONT_BOLD = "STSong-Light"
    _CJK_AVAILABLE = True
except Exception:
    import warnings
    warnings.warn(
        "CJK font (STSong-Light) not available. "
        "Chinese/Japanese/Korean characters may not render correctly. "
        "Falling back to Helvetica.",
        stacklevel=1,
    )

_COLOR_HEADING = HexColor("#1a1a2e")
_COLOR_H1_ACCENT = HexColor("#16213e")
_COLOR_LINK = HexColor("#2563eb")
_COLOR_BLOCKQUOTE = HexColor("#4b5563")
_COLOR_BLOCKQUOTE_BAR = HexColor("#d1d5db")
_COLOR_CODE_BG = HexColor("#f3f4f6")
_COLOR_CODE_BORDER = HexColor("#e5e7eb")
_COLOR_BODY = HexColor("#1f2937")
_COLOR_FOOTNOTE = HexColor("#6b7280")

_BULLET_CHARS = ["\u2022", "\u25e6", "\u25aa"]


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    common = {"fontName": _CJK_FONT, "alignment": TA_LEFT, "textColor": _COLOR_BODY}
    styles: dict[str, ParagraphStyle] = {}

    styles["h1"] = ParagraphStyle(
        "MDH1", parent=base["Heading1"],
        fontName=_CJK_FONT_BOLD, fontSize=22, leading=28,
        spaceAfter=10, spaceBefore=18,
        textColor=_COLOR_H1_ACCENT, borderWidth=0,
        borderPadding=(0, 0, 4, 0), borderColor=_COLOR_H1_ACCENT,
    )
    styles["h2"] = ParagraphStyle(
        "MDH2", parent=base["Heading2"],
        fontName=_CJK_FONT_BOLD, fontSize=17, leading=23,
        spaceAfter=8, spaceBefore=14, textColor=_COLOR_HEADING,
    )
    styles["h3"] = ParagraphStyle(
        "MDH3", parent=base["Heading3"],
        fontName=_CJK_FONT_BOLD, fontSize=14, leading=20,
        spaceAfter=6, spaceBefore=10, textColor=_COLOR_HEADING,
    )
    styles["h4"] = ParagraphStyle(
        "MDH4", parent=base["Heading4"],
        fontName=_CJK_FONT_BOLD, fontSize=12, leading=18,
        spaceAfter=4, spaceBefore=8, textColor=_COLOR_HEADING,
    )
    styles["body"] = ParagraphStyle(
        "MDBody", parent=base["Normal"],
        fontSize=11, leading=18, spaceAfter=6, **common,
    )
    styles["code"] = ParagraphStyle(
        "MDCode", parent=base["Normal"],
        fontName="Courier", fontSize=8.5, leading=12,
        spaceAfter=2, spaceBefore=0,
        leftIndent=0, rightIndent=0,
        textColor=HexColor("#374151"),
    )
    styles["blockquote"] = ParagraphStyle(
        "MDBlockquote", parent=base["Normal"],
        fontSize=11, leading=18, spaceAfter=4,
        leftIndent=16, textColor=_COLOR_BLOCKQUOTE,
        fontName=_CJK_FONT,
    )
    styles["footnote"] = ParagraphStyle(
        "MDFootnote", parent=base["Normal"],
        fontSize=9, leading=14, spaceAfter=3,
        textColor=_COLOR_FOOTNOTE, fontName=_CJK_FONT,
    )

    for depth in range(4):
        indent = 14 + depth * 16
        bullet_indent = indent - 10
        styles[f"bullet_{depth}"] = ParagraphStyle(
            f"MDBullet{depth}", parent=base["Normal"],
            fontSize=11, leading=18, spaceAfter=3,
            leftIndent=indent, bulletIndent=bullet_indent,
            **common,
        )
        styles[f"ordered_{depth}"] = ParagraphStyle(
            f"MDOrdered{depth}", parent=base["Normal"],
            fontSize=11, leading=18, spaceAfter=3,
            leftIndent=indent, bulletIndent=bullet_indent,
            **common,
        )

    return styles


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_fmt(text: str) -> str:
    """Apply inline markdown formatting: bold, italic, code, links, strikethrough."""
    text = _escape_xml(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<strike>\1</strike>", text)
    text = re.sub(
        r"`(.+?)`",
        r'<font face="Courier" size="9" backColor="#f3f4f6"> \1 </font>',
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        rf'<a href="\2" color="{_COLOR_LINK}">\1</a>',
        text,
    )
    return text


def _bullet_depth(line: str) -> int:
    """Determine nesting depth from leading whitespace (2 or 4 spaces per level)."""
    stripped = line.lstrip()
    indent = len(line) - len(stripped)
    if indent >= 4:
        return min(indent // 4, 3)
    if indent >= 2:
        return min(indent // 2, 3)
    return 0


def _make_code_block(code: str) -> Table:
    """Wrap code text in a styled table cell to simulate a code block with background."""
    style = ParagraphStyle(
        "CodeInner", fontName="Courier", fontSize=8.5, leading=12,
        textColor=HexColor("#374151"),
    )
    para = Paragraph(code.replace("\n", "<br/>"), style)
    tbl = Table(
        [[para]],
        colWidths=["100%"],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _COLOR_CODE_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, _COLOR_CODE_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]),
    )
    return tbl


def _make_blockquote(text: str, styles: dict[str, ParagraphStyle]) -> Table:
    """Wrap text in a table with a left accent bar to simulate a blockquote."""
    para = Paragraph(_inline_fmt(text), styles["blockquote"])
    tbl = Table(
        [[para]],
        colWidths=["100%"],
        style=TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEBEFOREDECOR", (0, 0), (0, -1), 3, _COLOR_BLOCKQUOTE_BAR, 0, "butt"),
            ("LINEBEFORE", (0, 0), (0, -1), 3, _COLOR_BLOCKQUOTE_BAR),
        ]),
    )
    return tbl


def _is_table_separator(line: str) -> bool:
    """Check if a line is a markdown table separator like |---|---|."""
    return bool(re.match(r"^\s*\|[\s\-:|]+\|\s*$", line))


def _parse_table_row(line: str) -> list[str]:
    """Split a markdown table row into cell texts."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _make_table(header_row: list[str], data_rows: list[list[str]],
                styles: dict[str, ParagraphStyle]) -> Table:
    """Build a styled reportlab Table from parsed markdown table data."""
    header_style = ParagraphStyle(
        "TableHeader", fontName=_CJK_FONT_BOLD, fontSize=10, leading=15,
        textColor=HexColor("#1f2937"),
    )
    cell_style = ParagraphStyle(
        "TableCell", fontName=_CJK_FONT, fontSize=10, leading=15,
        textColor=_COLOR_BODY,
    )
    header = [Paragraph(_inline_fmt(c), header_style) for c in header_row]
    rows = [[Paragraph(_inline_fmt(c), cell_style) for c in row] for row in data_rows]
    all_rows = [header] + rows

    n_cols = len(header_row)
    col_width = (A4[0] - 44 * mm) / n_cols
    tbl = Table(all_rows, colWidths=[col_width] * n_cols)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f3f4f6")),
        ("FONTNAME", (0, 0), (-1, 0), _CJK_FONT_BOLD),
        ("GRID", (0, 0), (-1, -1), 0.5, _COLOR_CODE_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl


def _parse_md(md_text: str, styles: dict[str, ParagraphStyle]) -> list:
    flowables: list = []
    lines = md_text.split("\n")
    i = 0
    in_code = False
    code_buf: list[str] = []
    ordered_counters: dict[int, int] = {}

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Code blocks ──────────────────────────────────────
        if stripped.startswith("```"):
            if in_code:
                code = _escape_xml("\n".join(code_buf))
                flowables.append(_make_code_block(code))
                flowables.append(Spacer(1, 3 * mm))
                code_buf.clear()
            in_code = not in_code
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # ── Empty line ────────────────────────────────────────
        if not stripped:
            ordered_counters.clear()
            flowables.append(Spacer(1, 2 * mm))
            i += 1
            continue

        # ── Horizontal rule ───────────────────────────────────
        if stripped in ("---", "***", "___"):
            flowables.append(Spacer(1, 2 * mm))
            flowables.append(HRFlowable(
                width="100%", thickness=0.5, color=_COLOR_CODE_BORDER,
                spaceBefore=2, spaceAfter=2,
            ))
            flowables.append(Spacer(1, 2 * mm))
            i += 1
            continue

        # ── Headings ──────────────────────────────────────────
        if stripped.startswith("#### "):
            flowables.append(Paragraph(_inline_fmt(stripped[5:]), styles["h4"]))
            i += 1
            continue
        if stripped.startswith("### "):
            flowables.append(Paragraph(_inline_fmt(stripped[4:]), styles["h3"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flowables.append(Paragraph(_inline_fmt(stripped[3:]), styles["h2"]))
            i += 1
            continue
        if stripped.startswith("# "):
            flowables.append(Paragraph(_inline_fmt(stripped[2:]), styles["h1"]))
            i += 1
            continue

        # ── Blockquote ────────────────────────────────────────
        if stripped.startswith("> "):
            quote_lines = [stripped[2:]]
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("> "):
                i += 1
                quote_lines.append(lines[i].strip()[2:])
            flowables.append(_make_blockquote("<br/>".join(quote_lines), styles))
            flowables.append(Spacer(1, 2 * mm))
            i += 1
            continue

        # ── Markdown table ───────────────────────────────────
        if "|" in stripped and not stripped.startswith("```"):
            table_lines = [line]
            j = i + 1
            while j < len(lines) and "|" in lines[j].strip():
                table_lines.append(lines[j])
                j += 1
            if len(table_lines) >= 2 and _is_table_separator(table_lines[1]):
                header = _parse_table_row(table_lines[0])
                data = [_parse_table_row(r) for r in table_lines[2:]
                        if not _is_table_separator(r)]
                # Normalize column counts
                n_cols = len(header)
                data = [row + [""] * (n_cols - len(row)) if len(row) < n_cols
                        else row[:n_cols] for row in data]
                flowables.append(_make_table(header, data, styles))
                flowables.append(Spacer(1, 3 * mm))
                i = j
                continue

        # ── Ordered list ──────────────────────────────────────
        m_ordered = re.match(r"^(\s*)(\d+)[.)]\s+(.+)", line)
        if m_ordered:
            depth = _bullet_depth(line)
            ordered_counters[depth] = ordered_counters.get(depth, 0) + 1
            num = ordered_counters[depth]
            text = _inline_fmt(m_ordered.group(3))
            style_key = f"ordered_{min(depth, 3)}"
            flowables.append(Paragraph(text, styles[style_key], bulletText=f"{num}."))
            i += 1
            continue

        # ── Unordered list ────────────────────────────────────
        m_bullet = re.match(r"^(\s*)[-*+]\s+(.+)", line)
        if m_bullet:
            depth = _bullet_depth(line)
            char = _BULLET_CHARS[min(depth, len(_BULLET_CHARS) - 1)]
            text = _inline_fmt(m_bullet.group(2))
            style_key = f"bullet_{min(depth, 3)}"
            flowables.append(Paragraph(text, styles[style_key], bulletText=char))
            i += 1
            continue

        # ── Footnote-style italic line ────────────────────────
        if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            flowables.append(Paragraph(_inline_fmt(stripped), styles["footnote"]))
            i += 1
            continue

        # ── Body text ─────────────────────────────────────────
        flowables.append(Paragraph(_inline_fmt(stripped), styles["body"]))
        i += 1

    return flowables


def convert(input_path: str, output_path: str) -> None:
    try:
        md_text = Path(input_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: Failed to read '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)

    styles = _build_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=25 * mm,
        bottomMargin=25 * mm,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
    )

    flowables = _parse_md(md_text, styles)
    if not flowables:
        flowables = [Paragraph("(empty document)", styles["body"])]

    try:
        doc.build(flowables)
    except Exception as e:
        print(f"Error: Failed to build PDF '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"SUCCESS: PDF created at {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(__file__).name} <input.md> <output.pdf>", file=sys.stderr)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
