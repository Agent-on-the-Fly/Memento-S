"""Block — user input 级别的事件容器。

每个 user input 对应一个 block，block 包含:
  - block_meta.json: 结构化元数据
  - events.jsonl: 该 block 内的事件流（一行一个 event）
  - artifacts/: compact 时被 fold 的长 tool_result 原文

目录结构:
  {context_dir}/sessions/{session_id}/
    session_meta.json
    runtime_state.json
    blocks/
      block_0001/
        block_meta.json
        events.jsonl
        artifacts/
          e0003.txt
          e0007.txt
      block_0002/
        ...
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Event
# ═══════════════════════════════════════════════════════════════


_event_counter: int = 0


def _next_event_id() -> str:
    global _event_counter
    _event_counter += 1
    return f"e{_event_counter:04d}"


def make_event(
    event_type: str,
    *,
    text: str = "",
    tool_name: str = "",
    args_summary: str = "",
    ref: str = "",
    preview: str = "",
    status: str = "",
    size: int = 0,
    tags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造一个标准 event dict。

    Event types:
      - user: 用户输入
      - plan: 计划步骤
      - tool_call: 工具调用
      - tool_result: 短结果（内联）
      - tool_result_ref: 长结果（已 fold 到 artifact）
      - action_result: 动作效果判定（effective/ineffective）
      - system: 系统事件（seal、compact 等）
    """
    event: dict[str, Any] = {
        "event_id": _next_event_id(),
        "type": event_type,
        "ts": time.time(),
    }
    if text:
        event["text"] = text
    if tool_name:
        event["tool_name"] = tool_name
    if args_summary:
        event["args_summary"] = args_summary
    if ref:
        event["ref"] = ref
    if preview:
        event["preview"] = preview
    if status:
        event["status"] = status
    if size:
        event["size"] = size
    if tags:
        event["tags"] = tags
    if extra:
        event.update(extra)
    return event


# ═══════════════════════════════════════════════════════════════
# BlockMeta
# ═══════════════════════════════════════════════════════════════


@dataclass
class BlockMeta:
    """一个 block 的结构化元数据。"""

    block_id: str = ""
    session_id: str = ""
    user_input_text: str = ""
    created_at: str = ""
    status: str = "active"  # active | sealed
    turn_count: int = 0
    event_count: int = 0
    kept_event_count: int = 0
    folded_chunk_count: int = 0
    artifact_count: int = 0
    tags: list[str] = field(default_factory=list)
    result_refs: list[str] = field(default_factory=list)
    superseded_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "session_id": self.session_id,
            "user_input_text": self.user_input_text,
            "created_at": self.created_at,
            "status": self.status,
            "turn_count": self.turn_count,
            "event_count": self.event_count,
            "kept_event_count": self.kept_event_count,
            "folded_chunk_count": self.folded_chunk_count,
            "artifact_count": self.artifact_count,
            "tags": self.tags,
            "result_refs": self.result_refs,
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockMeta":
        return cls(
            block_id=data.get("block_id", ""),
            session_id=data.get("session_id", ""),
            user_input_text=data.get("user_input_text", ""),
            created_at=data.get("created_at", ""),
            status=data.get("status", "active"),
            turn_count=data.get("turn_count", 0),
            event_count=data.get("event_count", 0),
            kept_event_count=data.get("kept_event_count", 0),
            folded_chunk_count=data.get("folded_chunk_count", 0),
            artifact_count=data.get("artifact_count", 0),
            tags=data.get("tags", []),
            result_refs=data.get("result_refs", []),
            superseded_by=data.get("superseded_by"),
        )


# ═══════════════════════════════════════════════════════════════
# Block
# ═══════════════════════════════════════════════════════════════


class Block:
    """单个 block 的操作接口。

    Manages:
      - block_meta.json
      - events.jsonl
    """

    def __init__(self, block_dir: Path, meta: BlockMeta) -> None:
        self._dir = block_dir
        self.meta = meta
        self._events_path = block_dir / "events.jsonl"
        self._meta_path = block_dir / "block_meta.json"

    @property
    def block_dir(self) -> Path:
        return self._dir

    @property
    def block_id(self) -> str:
        return self.meta.block_id

    def append_event(self, event: dict[str, Any]) -> None:
        """追加一个 event 到 events.jsonl。"""
        try:
            with open(self._events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.meta.event_count += 1
        except OSError:
            logger.warning("Failed to append event: {}", self._events_path, exc_info=True)

    def persist_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
        *,
        fold_char_limit: int = 4000,
        fold_line_limit: int = 120,
        preview_max_chars: int = 500,
        preview_max_lines: int = 5,
        status: str = "",
    ) -> dict[str, Any]:
        """统一处理 tool result：fold 判定 + event 记录 + 返回 tool_msg。

        短内容 → 记录 tool_result event，返回内联 tool_msg
        长内容 → 存 artifact，记录 tool_result_ref event，返回 ref+preview tool_msg

        Returns:
            LLM 格式的 tool message dict。
        """
        should_fold = (
            len(result) > fold_char_limit
            or result.count("\n") + 1 > fold_line_limit
        )

        if not should_fold:
            self.append_event(make_event(
                "tool_result",
                tool_name=tool_name,
                text=result,
                status=status,
                extra={"tool_call_id": tool_call_id},
            ))
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            }

        # 长内容：存 artifact，记录 ref event
        artifacts_dir = self._dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.meta.artifact_count += 1
        filename = f"artifact_{self.meta.artifact_count:04d}.md"
        artifact_path = artifacts_dir / filename
        ref_path = f"artifacts/{filename}"

        header = (
            f"# Artifact: {tool_name}\n"
            f"> created: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"> size: {len(result)} chars\n\n"
        )
        try:
            artifact_path.write_text(header + result, encoding="utf-8")
        except OSError:
            logger.warning(
                "Failed to write artifact: {}", artifact_path, exc_info=True,
            )
            self.append_event(make_event(
                "tool_result",
                tool_name=tool_name,
                text=result,
                status=status,
                extra={"tool_call_id": tool_call_id},
            ))
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            }

        # preview
        lines = result.split("\n")[:preview_max_lines]
        preview = "\n".join(lines)
        if len(preview) > preview_max_chars:
            preview = preview[:preview_max_chars]

        self.append_event(make_event(
            "tool_result_ref",
            tool_name=tool_name,
            ref=ref_path,
            preview=preview[:200],
            size=len(result),
            status=status,
            extra={"tool_call_id": tool_call_id},
        ))

        # 追踪 artifact ref（返回绝对路径供 runtime_state 使用）
        artifact_abs_path = str(artifact_path)

        folded_content = (
            f"[artifact_ref: {ref_path}]\n"
            f"{preview}\n"
            f"[{len(result)} chars, full content archived — "
            f"use read_file(path=\"{artifact_abs_path}\") to retrieve full text]"
        )
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": folded_content,
            "_artifact_ref": artifact_abs_path,
        }

    def load_events(self) -> list[dict[str, Any]]:
        """读取所有 events。"""
        if not self._events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            with open(self._events_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to load events: {}", self._events_path, exc_info=True)
        return events

    def load_recent_events(self, k: int = 8) -> list[dict[str, Any]]:
        """读取最近 k 轮的 events。

        轮次定义: 每个 assistant 事件开启一个新轮次。
        如果总轮次 <= k，返回所有 events。
        """
        all_events = self.load_events()

        round_starts = self._find_round_boundaries(all_events)

        if len(round_starts) <= k:
            return all_events

        split_idx = round_starts[-k]
        return all_events[split_idx:]

    def save_meta(self) -> None:
        """写入 block_meta.json。"""
        try:
            self._meta_path.write_text(
                json.dumps(self.meta.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to save block meta: {}", self._meta_path, exc_info=True)

    @staticmethod
    def _find_round_boundaries(events: list[dict[str, Any]]) -> list[int]:
        """找到事件列表中的轮次边界。

        每个 assistant 事件开启一个新轮次（包含后续的 tool_call / tool_result 等）。

        Returns:
            assistant 事件的索引列表，按升序排列。
        """
        return [i for i, ev in enumerate(events) if ev.get("type") == "assistant"]

    @staticmethod
    def _slim_tool_result_event(ev: dict[str, Any], max_chars: int = 120) -> dict[str, Any]:
        """将 tool_result event 的 text 截断为一行摘要，保留结构（无 artifact 落盘）。

        仅用于跨 block 回溯等不需要保留原文的场景。
        block 内 compact 请使用 _fold_tool_result_event()。

        tool_result_ref 本身已经是折叠态（ref+preview），不需要再压缩。
        其他类型的 event (user_input, assistant, tool_call, plan) 原样保留。
        """
        if ev.get("type") != "tool_result":
            return ev
        slimmed = dict(ev)
        text = slimmed.get("text", "")
        if len(text) > max_chars:
            slimmed["text"] = text[:max_chars] + "…"
            slimmed["_trimmed_from"] = len(text)
        return slimmed

    def _fold_tool_result_event(
        self, ev: dict[str, Any], max_chars: int = 120,
    ) -> dict[str, Any]:
        """将 tool_result event 折叠为 tool_result_ref，原文存入 artifacts/ 目录。

        - tool_result 且 text 长度 > max_chars → 原文落盘，转为 tool_result_ref
        - tool_result 且 text 长度 <= max_chars → 原样保留（无需折叠）
        - tool_result_ref / 其他类型 → 原样保留
        """
        if ev.get("type") != "tool_result":
            return ev
        text = ev.get("text", "")
        if len(text) <= max_chars:
            return ev

        # ── 原文落盘到 block artifacts/ 目录 ──
        artifacts_dir = self._dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        event_id = ev.get("event_id", "unknown")
        tool_name = ev.get("tool_name", "unknown")
        filename = f"{event_id}.txt"
        artifact_path = artifacts_dir / filename

        header = (
            f"# Artifact: {tool_name}\n"
            f"> event_id: {event_id}\n"
            f"> size: {len(text)} chars\n\n"
        )
        try:
            artifact_path.write_text(header + text, encoding="utf-8")
        except OSError:
            logger.warning(
                "Failed to write artifact: {}", artifact_path, exc_info=True,
            )
            # 降级为硬截断，不丢失事件
            slimmed = dict(ev)
            slimmed["text"] = text[:max_chars] + "…"
            slimmed["_trimmed_from"] = len(text)
            return slimmed

        # ── 构造 tool_result_ref 事件替换原 tool_result ──
        ref_event: dict[str, Any] = {
            "event_id": event_id,
            "type": "tool_result_ref",
            "ts": ev.get("ts", 0),
            "tool_name": tool_name,
            "ref": f"artifacts/{filename}",
            "preview": text[:max_chars] + "…",
            "size": len(text),
        }
        # 保留原 event 中的其他字段（如 status, tags 等）
        for key in ("status", "tags", "args_summary"):
            if key in ev:
                ref_event[key] = ev[key]

        self.meta.artifact_count += 1
        return ref_event

    def compact_old_events(
        self,
        keep_recent: int = 8,
        slim_max_chars: int = 120,
    ) -> dict[str, Any] | None:
        """折叠旧 events：按轮次边界保留最近 keep_recent 轮，将更早轮次的 tool_result 折叠为 artifact。

        轮次定义: 每个 assistant 事件开启一个新轮次，包含后续的 tool_call / tool_result 等。

        策略:
          - 最近 keep_recent 轮的所有 events → 原文保留
          - 更早轮次: user_input, assistant, tool_call, plan → 原样保留
          - 更早轮次: tool_result → 原文存入 artifacts/，转为 tool_result_ref（ref+preview）
          - tool_result_ref → 已是 ref+preview，原样保留

        Returns:
            stats dict (if compaction happened), or None.
        """
        all_events = self.load_events()

        # 按轮次（assistant 事件）确定边界
        round_starts = self._find_round_boundaries(all_events)

        if len(round_starts) <= keep_recent:
            return None  # 轮次不够，不需要折叠

        # 分割点：倒数第 keep_recent 个轮次的起始位置
        split_idx = round_starts[-keep_recent]
        old_events = all_events[:split_idx]
        kept_events = all_events[split_idx:]

        # ── Fold old events: tool_result 原文存入 artifacts/，转为 tool_result_ref ──
        folded_old: list[dict[str, Any]] = []
        tool_results_folded = 0
        artifact_refs: list[str] = []

        for ev in old_events:
            folded = self._fold_tool_result_event(ev, max_chars=slim_max_chars)
            if folded is not ev:  # was folded
                tool_results_folded += 1
            folded_old.append(folded)
            # Collect artifact refs for stats
            if folded.get("type") == "tool_result_ref":
                ref = folded.get("ref", "")
                if ref:
                    artifact_refs.append(ref)

        # ── 重写 events.jsonl: folded old + kept recent (原文) ──
        try:
            with open(self._events_path, "w", encoding="utf-8") as f:
                for ev in folded_old:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                for ev in kept_events:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning(
                "Failed to rewrite events after compact: {}",
                self._events_path, exc_info=True,
            )
            return None

        # ── 更新 meta ──
        new_total = len(folded_old) + len(kept_events)
        self.meta.kept_event_count = len(kept_events)
        self.meta.folded_chunk_count += 1
        self.meta.event_count = new_total
        self.save_meta()

        stats = {
            "old_event_count": len(old_events),
            "tool_results_folded": tool_results_folded,
            "artifact_refs": artifact_refs[:10],
            "kept_recent_rounds": keep_recent,
            "kept_recent_events": len(kept_events),
            "new_total": new_total,
        }

        logger.info(
            "Block compact: {} — folded {} tool results in {} old events, "
            "kept {} recent rounds ({} events)",
            self.meta.block_id,
            tool_results_folded,
            len(old_events),
            keep_recent,
            len(kept_events),
        )
        return stats

    def seal(self) -> None:
        """封存 block — 标记为 sealed，生成程序化的 block 摘要。"""
        all_events = self.load_events()

        # 统计 block 级别的摘要信息
        tool_names: list[str] = []
        artifact_refs: list[str] = []
        for ev in all_events:
            t = ev.get("type", "")
            if t == "tool_call":
                name = ev.get("tool_name", "")
                if name and name not in tool_names:
                    tool_names.append(name)
            elif t == "tool_result_ref":
                ref = ev.get("ref", "")
                if ref:
                    artifact_refs.append(ref)

        self.meta.status = "sealed"
        self.meta.kept_event_count = len(all_events)
        self.meta.result_refs = artifact_refs[:20]
        self.save_meta()

        logger.info(
            "Block sealed: {} ({} events, {} tools, {} artifacts)",
            self.meta.block_id,
            self.meta.event_count,
            len(tool_names),
            len(artifact_refs),
        )


# ═══════════════════════════════════════════════════════════════
# BlockManager
# ═══════════════════════════════════════════════════════════════


class BlockManager:
    """管理一个 session 的所有 blocks。

    目录: {context_dir}/sessions/{session_id}/blocks/
    """

    def __init__(self, session_id: str, session_dir: Path) -> None:
        self._session_id = session_id
        self._blocks_dir = session_dir / "blocks"
        self._blocks_dir.mkdir(parents=True, exist_ok=True)

        self._active_block: Block | None = None
        self._block_count: int = 0

        # Recover block count from existing directories
        self._block_count = sum(
            1 for d in self._blocks_dir.iterdir()
            if d.is_dir() and d.name.startswith("block_")
        )

        # Try to recover active block
        if self._block_count > 0:
            last_block_dir = self._blocks_dir / f"block_{self._block_count:04d}"
            meta_path = last_block_dir / "block_meta.json"
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta = BlockMeta.from_dict(data)
                    if meta.status == "active":
                        self._active_block = Block(last_block_dir, meta)
                        # Recover event counter
                        events = self._active_block.load_events()
                        if events:
                            last_eid = events[-1].get("event_id", "e0000")
                            global _event_counter
                            try:
                                _event_counter = max(
                                    _event_counter,
                                    int(last_eid.lstrip("e")),
                                )
                            except ValueError:
                                pass
                except (OSError, json.JSONDecodeError):
                    logger.warning("Failed to recover active block", exc_info=True)

    @property
    def active_block(self) -> Block | None:
        return self._active_block

    @property
    def block_count(self) -> int:
        return self._block_count

    def create_block(self, user_input_text: str) -> Block:
        """创建新 block。不会自动 seal 旧 block（调用方负责）。"""
        self._block_count += 1
        block_id = f"block_{self._block_count:04d}"
        block_dir = self._blocks_dir / block_id
        block_dir.mkdir(parents=True, exist_ok=True)

        meta = BlockMeta(
            block_id=block_id,
            session_id=self._session_id,
            user_input_text=user_input_text[:500],  # cap for meta
            created_at=datetime.now().isoformat(timespec="seconds"),
            status="active",
        )

        block = Block(block_dir, meta)

        # Write initial user event
        user_event = make_event("user", text=user_input_text)
        block.append_event(user_event)

        # Save meta
        block.save_meta()

        self._active_block = block
        logger.info("Block created: {} for session {}", block_id, self._session_id)
        return block

    def seal_active_block(self) -> Block | None:
        """封存当前 active block，返回被封存的 block（如果有的话）。"""
        if self._active_block is None:
            return None
        if self._active_block.meta.status == "sealed":
            return None

        sealed = self._active_block
        sealed.seal()
        self._active_block = None
        return sealed

    def get_block(self, block_id: str) -> Block | None:
        """按 ID 加载一个 block。"""
        block_dir = self._blocks_dir / block_id
        meta_path = block_dir / "block_meta.json"
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            return Block(block_dir, BlockMeta.from_dict(data))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to load block: {}", block_id, exc_info=True)
            return None

    def load_recent_sealed_events(
        self,
        past_blocks_k: int = 2,
        slim_max_chars: int = 120,
    ) -> list[dict[str, Any]]:
        """加载最近 past_blocks_k 个 sealed blocks 的事件（tool_result 已 slim）。

        用于跨 block 上下文传递：新 block 开始时，LLM 能看到前几轮的操作记录。
        tool_result_ref 本身已折叠，原样保留；tool_result 的 text 截断为摘要。

        Returns:
            合并后的 slimmed events 列表（按时间顺序）。
        """
        if past_blocks_k <= 0 or self._block_count <= 1:
            return []

        # 找 active block 之前的 sealed blocks
        active_id = self._active_block.block_id if self._active_block else None
        sealed_indices: list[int] = []
        for i in range(self._block_count, 0, -1):
            bid = f"block_{i:04d}"
            if bid == active_id:
                continue
            sealed_indices.append(i)
            if len(sealed_indices) >= past_blocks_k:
                break

        sealed_indices.reverse()  # 按时间顺序

        all_events: list[dict[str, Any]] = []
        for idx in sealed_indices:
            block = self.get_block(f"block_{idx:04d}")
            if block is None or block.meta.status != "sealed":
                continue
            for ev in block.load_events():
                slimmed = Block._slim_tool_result_event(ev, max_chars=slim_max_chars)
                all_events.append(slimmed)

        return all_events

    def list_block_metas(self) -> list[BlockMeta]:
        """列出所有 block 的 meta（按创建顺序）。"""
        metas: list[BlockMeta] = []
        for i in range(1, self._block_count + 1):
            block_id = f"block_{i:04d}"
            block_dir = self._blocks_dir / block_id
            meta_path = block_dir / "block_meta.json"
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                    metas.append(BlockMeta.from_dict(data))
                except (OSError, json.JSONDecodeError):
                    pass
        return metas


# ═══════════════════════════════════════════════════════════════
# SessionDir helper
# ═══════════════════════════════════════════════════════════════


def ensure_session_dir(context_dir: Path, session_id: str) -> Path:
    """确保 session 目录存在并返回路径。

    Returns:
        {context_dir}/sessions/{session_id}/
    """
    session_dir = context_dir / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir
