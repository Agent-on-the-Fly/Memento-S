# SPDX-License-Identifier: Apache-2.0
"""local_recall — 基于注册表的本地关键词召回

读取 SkillRegistry 获取本地 skill 列表，
在 name + description 上做关键词匹配。
无 embedding，无向量搜索。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from shared.schema import SkillConfig, SkillSearchResult

from core.skill.loader.skill_loader import SkillLoader
from core.skill.registry import registry as skill_registry
from core.skill.retrieval.base import BaseRecall
from utils.strings import to_kebab_case
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _CacheEntry:
    """单个 skill 的缓存条目"""
    skill_name: str
    description: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    status: str = "active"
    mtime: float = 0.0
    size: int = 0


@dataclass
class _ScanState:
    """扫描状态缓存"""
    entries: dict[str, _CacheEntry] = field(default_factory=dict)
    dir_signature: str = ""
    last_scan_time: float = 0.0


class LocalRecall(BaseRecall):
    """基于注册表的本地关键词召回

    读取 SkillRegistry 获取本地 skill 列表，
    在 name + description 上做关键词匹配。
    无 embedding，无向量搜索。

    缓存策略：
    - mtime/size 指纹验证（不变时不重新读文件）
    - 内存中的 score 缓存（下次 search 直接返回）
    """

    def __init__(self, skills_dir: Path | str) -> None:
        self._skills_dir = Path(skills_dir)
        self._registry = skill_registry
        self._state = _ScanState()
    @classmethod
    def from_config(cls, config: SkillConfig) -> "LocalRecall":
        return cls(config.skills_dir)

    @property
    def name(self) -> str:
        return "local"

    def is_available(self) -> bool:
        return self._skills_dir.exists() and self._skills_dir.is_dir()

    # ── 缓存 ────────────────────────────────────────────────────────────────

    def _compute_dir_signature(self) -> str:
        """计算目录状态签名（MD5），用于快速检测变更"""
        sig_parts: list[str] = []
        if not self._skills_dir.exists():
            return hashlib.md5(b"").hexdigest()
        for item in sorted(self._skills_dir.iterdir()):
            if not item.is_dir():
                continue
            skill_md = item / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                stat = skill_md.stat()
                sig_parts.append(f"{item.name}:{stat.st_mtime:.6f}:{stat.st_size}")
            except OSError:
                continue
        return hashlib.md5("|".join(sig_parts).encode()).hexdigest()

    def _has_changes(self) -> bool:
        return self._compute_dir_signature() != self._state.dir_signature

    # ── 扫描 ────────────────────────────────────────────────────────────────

    def _scan_directory(self) -> dict[str, _CacheEntry]:
        """扫描 skills 目录，构建缓存条目（轻量级，只读 frontmatter）"""
        result: dict[str, _CacheEntry] = {}
        if not self._skills_dir.exists():
            return result

        for item in sorted(self._skills_dir.iterdir()):
            if not item.is_dir():
                continue
            skill_md = item / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                stat = skill_md.stat()
                mtime = stat.st_mtime
                size = stat.st_size
                name_kebab = item.name

                # 检查缓存是否有效
                if name_kebab in self._state.entries:
                    entry = self._state.entries[name_kebab]
                    if entry.mtime == mtime and entry.size == size:
                        result[name_kebab] = entry
                        continue

                # 重新加载（轻量级）
                fm = self._load_frontmatter(item)
                result[name_kebab] = _CacheEntry(
                    skill_name=name_kebab,
                    description=fm.get("description", "") or "",
                    keywords=fm.get("keywords", []),
                    tags=fm.get("tags", []),
                    mtime=mtime,
                    size=size,
                )
            except OSError:
                continue

        return result

    def _load_description(self, skill_dir: Path) -> str:
        """从 SKILL.md 加载 description（轻量级）"""
        skill_md = skill_dir / "SKILL.md"
        try:
            import frontmatter
            post = frontmatter.load(str(skill_md))
            return str(post.metadata.get("description", "") or "")
        except Exception:
            return ""

    def _load_frontmatter(self, skill_dir: Path) -> dict[str, Any]:
        """从 SKILL.md 加载完整 frontmatter 元数据"""
        skill_md = skill_dir / "SKILL.md"
        try:
            import frontmatter
            post = frontmatter.load(str(skill_md))
            return {
                "description": str(post.metadata.get("description", "") or ""),
                "keywords": self._parse_keywords(post.metadata.get("keywords", "")),
                "tags": self._parse_tags(post.metadata.get("tags", "")),
            }
        except Exception:
            return {"description": "", "keywords": [], "tags": []}

    def _parse_keywords(self, value: Any) -> list[str]:
        """解析 keywords 字段，支持字符串或列表"""
        if isinstance(value, list):
            return [str(k).lower().strip() for k in value if k]
        if isinstance(value, str):
            return [k.strip().lower() for k in value.split(",") if k.strip()]
        return []

    def _parse_tags(self, value: Any) -> list[str]:
        """解析 tags 字段，支持字符串或列表"""
        if isinstance(value, list):
            return [str(t).lower().strip() for t in value if t]
        if isinstance(value, str):
            return [t.strip().lower() for t in value.split(",") if t.strip()]
        return []

    def _refresh_cache(self) -> None:
        """刷新缓存"""
        start = time.time()
        new_entries = self._scan_directory()
        self._state.entries = new_entries
        self._state.dir_signature = self._compute_dir_signature()
        self._state.last_scan_time = time.time()
        logger.debug(
            "[LOCAL_RECALL] Cache refreshed: {} skills in {:.1f}ms",
            len(new_entries),
            (time.time() - start) * 1000,
        )

    # ── 关键词匹配打分 ────────────────────────────────────────────────────────

    def _match_score(
        self,
        query: str,
        name: str,
        description: str,
        keywords: list[str],
        tags: list[str],
    ) -> float:
        """关键词匹配打分。

        检索策略（无 embedding，纯关键词匹配）：

        | 匹配类型                     | 分数  |
        |-----------------------------|-------|
        | token 完全匹配 name         | 1.0   |
        | token 完全匹配 keywords     | 0.9   |
        | token 是 name 子串          | 0.7   |
        | token 完全匹配 description  | 0.5   |
        | token 匹配 tags             | 0.6   |
        | query 整体是子串             | 0.3   |
        | fuzzy match (edit distance) | 0.3 * similarity |
        | 无匹配                      | 0.0   |

        最终取最高分。前缀匹配额外加分（name 前缀 +0.1, desc 前缀 +0.05）。

        Returns:
            最高匹配分数 (0.0 ~ 1.0)
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return 1.0  # 无 query，返回所有

        tokens = query_lower.split()
        name_lower = name.lower()
        desc_lower = description.lower()
        keyword_strs = [k.lower() for k in keywords]
        tag_strs = [t.lower() for t in tags]

        max_score = 0.0

        for token in tokens:
            # 1. token 全匹配 name（前缀匹配视为 1.0）
            if token == name_lower or name_lower.startswith(token):
                max_score = max(max_score, 1.0)
                continue

            # 2. token 全匹配 keywords
            if token in keyword_strs:
                max_score = max(max_score, 0.9)
                continue

            # 3. token 匹配 tags
            if token in tag_strs:
                max_score = max(max_score, 0.6)
                continue

            # 4. token 是 name 子串
            if token in name_lower:
                max_score = max(max_score, 0.7)
                continue

            # 5. token 完全匹配 description
            if token in desc_lower:
                max_score = max(max_score, 0.5)
                continue

            # 6. fuzzy matching（编辑距离）
            for target in [name_lower, desc_lower] + keyword_strs + tag_strs:
                if not target:
                    continue
                fuzzy = SequenceMatcher(None, token, target).ratio()
                if fuzzy >= 0.6:  # 阈值 0.6
                    max_score = max(max_score, 0.3 * fuzzy)
                    break

            # 7. query 整体作为子串出现
            if query_lower in name_lower or query_lower in desc_lower:
                max_score = max(max_score, 0.3)
                continue

        return max(0.0, min(1.0, max_score))

    # ── 搜索 ────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        k: int = 5,
        locations: list[str] | None = None,
        status_filter: str = "active",
        **kwargs,
    ) -> list[RecallCandidate]:
        """关键词搜索本地 skills。

        检索策略：
        1. 验证缓存（mtime/size fingerprint）
        2. 从 SkillRegistry 获取所有注册 skill（包含 absolute path location）
        3. 按 location 前缀/status 过滤
        4. 对每个 skill 计算关键词匹配 score
        5. 按 score 降序，取前 k 个（score > 0）
        6. 保留 full skill 对象供 gateway 使用

        Args:
            query: 搜索查询
            k: 返回的最大结果数
            locations: 可选，按 skill 目录绝对路径前缀过滤，
                如 ["/path/to/skills/", "/path/to/builtin/"]
            status_filter: 按 status 过滤，默认 "active"
        """
        # 确保缓存最新
        if self._has_changes():
            self._refresh_cache()

        # 获取注册表中的所有 skill（包含 source 信息）
        self._registry.load()
        registry_skills = self._registry.list_all()
        # 统一注册表 key 为 kebab-case，确保与 _state.entries key 格式一致
        registry_skills_normalized = {
            to_kebab_case(k): v for k, v in registry_skills.items()
        }

        # 合并两侧 name，统一转为 kebab-case
        all_names: set[str] = set(self._state.entries.keys()) | set(registry_skills_normalized.keys())

        candidates: list[SkillSearchResult] = []
        for name in all_names:
            entry = self._state.entries.get(name)
            registry_meta = registry_skills_normalized.get(name)

            skill_name = entry.skill_name if entry else name
            description = entry.description if entry else (
                registry_meta.get("description", "") if registry_meta else ""
            )
            keywords = entry.keywords if entry else []
            tags = entry.tags if entry else []

            # 过滤：按 location（registry_location 是绝对路径，locations 是路径前缀列表）
            if locations and registry_meta:
                registry_location = registry_meta.get("location", "")
                if not any(registry_location.startswith(prefix) for prefix in locations):
                    continue

            # 过滤：按 status（只对已在注册表的 skill 检查 registry status，
            # 不在注册表的 skill 不做 status 过滤，默认视为 active）
            if registry_meta:
                registry_status = registry_meta.get("status", "active")
                if registry_status != status_filter:
                    continue

            score = self._match_score(query, skill_name, description, keywords, tags)
            if score <= 0.0:
                continue

            # 确定 source（从 registry 读取），统一映射到 "local" / "cloud"
            raw_source = registry_meta.get("source", "local") if registry_meta else "local"
            source = "local" if raw_source != "cloud" else "cloud"

            candidates.append(SkillSearchResult(
                name=skill_name,
                description=description,
                source=source,
                score=score,
                match_type="keyword",
            ))

        # 按 score 降序
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:k]

    # ── 工具 ────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats.update({
            "skills_dir": str(self._skills_dir),
            "cached_skills": len(self._state.entries),
            "last_scan_time": self._state.last_scan_time,
        })
        return stats


# ── 独立加载函数（供 gateway._ensure_local_skill 使用） ──────────────────────

def load_full_skill(skills_dir: Path | str, name: str) -> Skill | None:
    """按名称从 skills_dir 加载完整 skill（含 scripts/references）。

    内部委托给 SkillLoader.load()，兼容层保留此函数。
    """
    loader = SkillLoader(skills_dir)
    return loader.load(name) or loader.load(to_kebab_case(name).replace("-", "_"))
