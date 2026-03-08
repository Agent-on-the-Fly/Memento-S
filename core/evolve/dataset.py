"""Dataset loading — HLE rows, coercion, user text construction, and attachments."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .text_utils import dedupe_keep_order, normalize_space


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _clean_text_field(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    return str(raw).strip()


def _first_text_field(row: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        text = _clean_text_field(v)
        if text:
            return text
    return ""


def _looks_like_image_ref(ref: str) -> bool:
    low = str(ref or "").strip().lower()
    if low.startswith("data:image/"):
        return True
    return any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"))


def _stringify_image_ref(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, bytes):
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    return ""


def _extract_attachment_refs(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        return [text]
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            text = _clean_text_field(item)
            if text:
                out.append(text)
        return out
    return []


def _collect_attachment_refs(row: dict) -> list[str]:
    refs: list[str] = []
    for key in ("attachments", "attachment", "files", "file_paths"):
        val = row.get(key)
        extracted = _extract_attachment_refs(val)
        refs.extend(extracted)
    # Also collect file_path / file_name (GAIA-style) and image_path (HLE-style)
    for key in ("file_path", "image_path"):
        val = _clean_text_field(row.get(key))
        if val and val not in refs:
            refs.append(val)
    return dedupe_keep_order(refs)


# ---------------------------------------------------------------------------
# Row coercion
# ---------------------------------------------------------------------------

def coerce_hle_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize raw dataset rows to a canonical schema."""
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        q = _first_text_field(row, ("question", "Question", "prompt", "Prompt", "instruction", "Instruction"))
        a = _first_text_field(
            row,
            ("answer", "Answer", "final_answer", "Final answer", "gold_answer", "gold", "target", "label"),
        )
        if not q or not a:
            continue
        # Prefer file-path reference over inline base64 — cheaper and
        # the agent can view the file directly.
        image_ref = _clean_text_field(row.get("image_path"))
        if not image_ref:
            image_ref = _stringify_image_ref(row.get("image"))
        attachments = _collect_attachment_refs(row)
        if image_ref and image_ref not in attachments:
            attachments.insert(0, image_ref)
        if not image_ref:
            for ref in attachments:
                if _looks_like_image_ref(ref):
                    image_ref = ref
                    break
        task_id = _first_text_field(row, ("id", "task_id", "taskId", "uid")) or f"idx:{idx}"
        out.append(
            {
                "id": task_id,
                "question": q,
                "answer": a,
                "image": image_ref,
                "attachments": attachments,
                "answer_type": _first_text_field(row, ("answer_type", "answerType", "Answer type")),
                "category": _first_text_field(row, ("category", "Category")),
                "raw_subject": _first_text_field(row, ("raw_subject", "subject", "Subject")),
                "level": _first_text_field(row, ("level", "Level")),
                "rationale": _first_text_field(row, ("rationale", "Rationale")),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_hle_rows(path: Path) -> list[dict[str, Any]]:
    """Load HLE rows from .json, .jsonl, or .parquet."""
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        if suffix == ".json":
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                return []
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("JSON dataset must be a list of objects")
            return coerce_hle_rows([row for row in parsed if isinstance(row, dict)])
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    preview = line[:220].replace("\n", "\\n")
                    raise ValueError(
                        f"Invalid JSONL at line {lineno}: {exc.msg} (col={exc.colno}) preview={preview!r}"
                    ) from exc
                if isinstance(obj, dict):
                    rows.append(obj)
        return coerce_hle_rows(rows)

    if suffix == ".parquet":
        pq_rows: list[dict[str, Any]] | None = None
        pyarrow_err: Exception | None = None
        try:
            import pyarrow.parquet as pq  # type: ignore[import-untyped]
            table = pq.read_table(path)
            pq_rows = [r for r in table.to_pylist() if isinstance(r, dict)]
        except Exception as exc:
            pyarrow_err = exc

        if pq_rows is None:
            datasets_err: Exception | None = None
            try:
                from datasets import load_dataset  # type: ignore[import-untyped]
                ds = load_dataset("parquet", data_files={"test": str(path)}, split="test")
                pq_rows = [dict(item) for item in ds]  # type: ignore[arg-type]
            except Exception as exc:
                datasets_err = exc

            if pq_rows is None:
                raise RuntimeError(
                    "Cannot read parquet dataset. Install `pyarrow` or `datasets` first.\n"
                    f"pyarrow error: {pyarrow_err}\n"
                    f"datasets error: {datasets_err}"
                )
        return coerce_hle_rows(pq_rows)

    raise ValueError(f"Unsupported dataset format: {path}")


# ---------------------------------------------------------------------------
# Attachment handling
# ---------------------------------------------------------------------------

def _resolve_attachment_hint(attachment_ref: str, data_path: Path, project_root: Path) -> str:
    ref = _clean_text_field(attachment_ref)
    if not ref:
        return ""
    if ref.startswith("data:image/"):
        cache_dir = (data_path.parent / "image_cache").resolve()
        materialized = _materialize_data_uri_image(ref, cache_dir=cache_dir)
        if materialized:
            return materialized
    if re.match(r"^https?://", ref, re.IGNORECASE):
        return ref
    p = Path(ref)
    attachment_dirs = [
        (data_path.parent / "attachment").resolve(),
        (data_path.parent / "attachments").resolve(),
        (data_path.parent / "image_cache").resolve(),
        (data_path.parent.parent / "attachment").resolve(),
        (data_path.parent.parent / "attachments").resolve(),
        (data_path.parent.parent / "image_cache").resolve(),
        (project_root / "data" / "attachment").resolve(),
        (project_root / "data" / "attachments").resolve(),
        (project_root / "data" / "image_cache").resolve(),
        # GAIA-style: gaia_data/attachment/
        (project_root / "gaia_data" / "attachment").resolve(),
        # HLE-style: hle_data/image_cache/
        (project_root / "hle_data" / "image_cache").resolve(),
    ]
    if p.is_absolute():
        if p.exists():
            return str(p)
        candidates = [p]
        if p.name:
            candidates.extend([(base / p.name).resolve() for base in attachment_dirs])
    else:
        candidates = [
            (data_path.parent / ref).resolve(),
            (data_path.parent.parent / ref).resolve(),
            (project_root / ref).resolve(),
        ]
        # Handle paths like "Memento-S/hle_data/..." where the first
        # component matches project_root.name — strip it to avoid
        # doubling (project_root / "Memento-S" / "hle_data/...").
        root_prefix = project_root.name + "/"
        if ref.startswith(root_prefix):
            stripped = ref[len(root_prefix):]
            candidates.insert(0, (project_root / stripped).resolve())
        if p.name:
            candidates.extend([(base / p.name).resolve() for base in attachment_dirs])

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(c)

    for c in unique_candidates:
        if c.exists():
            return str(c)
    if unique_candidates:
        return str(unique_candidates[0])
    return ref


def _materialize_data_uri_image(data_uri: str, *, cache_dir: Path) -> str:
    text = str(data_uri or "").strip()
    m = re.match(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", text, re.DOTALL)
    if not m:
        return ""
    mime = m.group(1).lower()
    b64_data = re.sub(r"\s+", "", m.group(2))
    try:
        raw = base64.b64decode(b64_data, validate=False)
    except Exception:
        return ""
    if not raw:
        return ""
    ext_map = {
        "image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png",
        "image/webp": "webp", "image/gif": "gif", "image/bmp": "bmp",
        "image/tiff": "tiff", "image/svg+xml": "svg",
    }
    ext = ext_map.get(mime, "img")
    digest = hashlib.sha1(raw).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = (cache_dir / f"{digest}.{ext}").resolve()
    if not out_path.exists():
        out_path.write_bytes(raw)
    return str(out_path)


def _safe_task_dir_name(task_key: str) -> str:
    raw = str(task_key or "").strip()
    if not raw:
        return "task_unknown"
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)
    cleaned = cleaned.strip("._-")
    return cleaned or "task_unknown"


def _is_archive_file(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".zip") or any(
        lower.endswith(ext) for ext in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
    )


def prepare_task_attachments(
    *,
    task: dict[str, Any],
    data_path: Path,
    task_key: str,
    run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Resolve, copy, and extract task attachments into ``run_dir``."""
    image_ref = str(task.get("image") or "").strip()
    raw_attachments = _extract_attachment_refs(task.get("attachments"))
    if not raw_attachments:
        raw_attachments = _collect_attachment_refs(task)
    if image_ref and image_ref not in raw_attachments:
        raw_attachments.insert(0, image_ref)

    resolved_refs = dedupe_keep_order(
        [_resolve_attachment_hint(ref, data_path, project_root) for ref in raw_attachments if ref]
    )
    resolved_image_ref = (
        _resolve_attachment_hint(image_ref, data_path, project_root) if image_ref else ""
    )

    task_root = (run_dir / "task_attachments" / _safe_task_dir_name(task_key)).resolve()
    if task_root.exists():
        shutil.rmtree(task_root, ignore_errors=True)
    task_root.mkdir(parents=True, exist_ok=True)
    source_dir = (task_root / "source").resolve()
    extracted_dir = (task_root / "extracted").resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    src_to_outputs: dict[str, list[str]] = {}

    def _push(src: str, out: str) -> None:
        key = str(src or "").strip()
        val = str(out or "").strip()
        if not key or not val:
            return
        src_to_outputs.setdefault(key, []).append(val)

    for i, ref in enumerate(resolved_refs):
        ref_s = str(ref or "").strip()
        if not ref_s:
            continue
        if re.match(r"^https?://", ref_s, re.IGNORECASE):
            _push(ref_s, ref_s)
            continue
        p = Path(ref_s)
        if not p.exists():
            _push(ref_s, ref_s)
            continue
        if p.is_dir():
            dst = (source_dir / f"{i:03d}_{p.name}").resolve()
            shutil.copytree(p, dst, dirs_exist_ok=True)
            _push(ref_s, str(dst))
            continue
        dst_file = (source_dir / f"{i:03d}_{p.name}").resolve()
        shutil.copy2(p, dst_file)
        _push(ref_s, str(dst_file))
        if _is_archive_file(dst_file):
            extract_dst = (extracted_dir / f"{i:03d}_{p.stem}").resolve()
            extract_dst.mkdir(parents=True, exist_ok=True)
            try:
                shutil.unpack_archive(str(dst_file), str(extract_dst))
                _push(ref_s, str(extract_dst))
            except Exception:
                pass

    prepared_refs: list[str] = []
    for ref in resolved_refs:
        outputs = src_to_outputs.get(str(ref or "").strip(), [])
        if outputs:
            prepared_refs.extend(outputs)
        else:
            prepared_refs.append(str(ref or "").strip())
    prepared_refs = dedupe_keep_order([x for x in prepared_refs if str(x or "").strip()])

    prepared_image_ref = ""
    if resolved_image_ref:
        mapped = src_to_outputs.get(str(resolved_image_ref).strip(), [])
        if mapped:
            prepared_image_ref = mapped[0]
        else:
            prepared_image_ref = resolved_image_ref

    return {
        "task_root": str(task_root),
        "attachment_refs": prepared_refs,
        "image_ref": str(prepared_image_ref or ""),
        "resolved_original_refs": resolved_refs,
    }


# ---------------------------------------------------------------------------
# User text construction
# ---------------------------------------------------------------------------

def build_user_text(
    task: dict[str, Any],
    *,
    data_path: Path,
    project_root: Path,
    learning_tips: str = "",
    read_only_mode: bool = False,
    attachment_refs_override: list[str] | None = None,
    image_ref_override: str | None = None,
) -> str:
    question = str(task.get("question") or "").strip()
    image_ref = str(task.get("image") or "").strip()
    if attachment_refs_override is None:
        attachments = _extract_attachment_refs(task.get("attachments"))
        if not attachments:
            attachments = _collect_attachment_refs(task)
        if image_ref and image_ref not in attachments:
            attachments.insert(0, image_ref)
        resolved_attachments = dedupe_keep_order(
            [_resolve_attachment_hint(ref, data_path, project_root) for ref in attachments if ref]
        )
    else:
        resolved_attachments = dedupe_keep_order(
            [str(ref or "").strip() for ref in attachment_refs_override if str(ref or "").strip()]
        )

    if image_ref_override is None:
        resolved_image_ref = (
            _resolve_attachment_hint(image_ref, data_path, project_root) if image_ref else ""
        )
    else:
        resolved_image_ref = str(image_ref_override or "").strip()
    answer_type = str(task.get("answer_type") or "").strip()

    parts: list[str] = [question]
    if resolved_image_ref:
        parts.append("")
        parts.append("Image reference:")
        parts.append(f"- {resolved_image_ref}")
        parts.append("If visual content matters, use skills to inspect/process the image.")
    if resolved_attachments:
        parts.append("")
        parts.append("Attachment references:")
        for ref in resolved_attachments:
            parts.append(f"- {ref}")
        parts.append("If attachments matter, use suitable skills/tools to inspect or parse them before answering.")

    parts.append("")
    parts.append("Return exactly this format:")
    parts.append("Explanation: <brief reasoning>")
    if answer_type:
        parts.append(f"Answer: <final answer only; type={answer_type}>")
    else:
        parts.append("Answer: <final answer only>")
    parts.append("Confidence: <0-100>%")

    if learning_tips:
        parts.append("")
        parts.append("Learning Tips (from previous runs):")
        parts.append(learning_tips)

    if read_only_mode:
        parts.append("")
        parts.append("Read-only execution policy:")
        parts.append("- Use ONLY existing local skills; do NOT invent or fetch new skills.")
        parts.append("- If no existing skill is suitable, answer directly in the required format.")
        parts.append("- Never return raw tool output as the final answer.")

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------

def task_key(idx: int, task: dict[str, Any]) -> str:
    raw = str(task.get("id") or "").strip()
    return raw if raw else f"idx:{idx}"


def iter_tasks(tasks: list[dict[str, Any]], start: int, end: int | None, limit: int | None):
    yielded = 0
    for idx, task in enumerate(tasks):
        if idx < start:
            continue
        if end is not None and idx > end:
            break
        if limit is not None and yielded >= limit:
            break
        yielded += 1
        yield idx, task


def load_learning_tips(path: Path | None, max_chars: int) -> str:
    if path is None:
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n\n[Learning tips truncated]"
    return text
