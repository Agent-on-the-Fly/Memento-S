"""执行产出文件的发现、过滤、回传工具。

仅供 UvLocalSandbox 内部使用，不对外暴露。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from middleware.config import g_config
from utils.logger import get_logger

logger = get_logger(__name__)


class ArtifactManager:
    """执行产物管理工具类。

    此类仅供 UvLocalSandbox 内部使用，所有方法均为类方法。
    """

    _IGNORE_NAMES: set[str] = {
        "__runner__.py",
        "__params__.json",
    }
    _IGNORE_SUFFIXES: tuple[str, ...] = (
        ".pyc",
        "__pycache__",
    )

    @classmethod
    def should_ignore(cls, rel_path: str) -> bool:
        """检查文件是否应该被忽略。"""
        name = Path(rel_path).name
        if name in cls._IGNORE_NAMES:
            return True
        return any(name.endswith(suffix) for suffix in cls._IGNORE_SUFFIXES)

    @classmethod
    def get_sandbox_dir(cls, skill_name: str, session_id: str = "") -> Path:
        """获取技能执行的沙箱工作目录。

        Args:
            skill_name: 技能名称
            session_id: 会话ID（可选）

        Returns:
            沙箱目录路径
        """
        return g_config.get_session_sandbox_dir(skill_name, session_id)

    @classmethod
    def get_output_dir(cls, skill_name: str, session_id: str = "") -> Path:
        """获取技能执行产物输出目录。

        Args:
            skill_name: 技能名称
            session_id: 会话ID（可选）

        Returns:
            输出目录路径
        """
        return g_config.get_session_output_dir(skill_name, session_id)

    @classmethod
    def collect_local_artifacts(
        cls,
        work_dir: Path,
        pre_files: set[str],
        skill_name: str,
        session_id: str = "",
    ) -> list[str]:
        """收集本地执行产物。

        Args:
            work_dir: 工作目录
            pre_files: 执行前的文件集合
            skill_name: 技能名称
            session_id: 会话ID（可选）

        Returns:
            收集到的产物文件路径列表
        """
        post_files = cls.snapshot_files(work_dir)
        new_files = post_files - pre_files
        new_files = {f for f in new_files if not cls.should_ignore(f)}

        if not new_files:
            return []

        output_dir = cls.get_output_dir(skill_name, session_id=session_id)
        local_artifacts: list[str] = []

        for rel in sorted(new_files):
            src = work_dir / rel
            dst = output_dir / rel
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                local_artifacts.append(str(dst))
                logger.debug("Collected artifact: {} → {}", rel, dst)
            except Exception as e:
                logger.warning("Failed to collect artifact '{}': {}", rel, e)

        if local_artifacts:
            logger.info(
                f"Collected {len(local_artifacts)} artifacts for '{skill_name}' to {output_dir}"
            )

        return local_artifacts

    @classmethod
    def snapshot_files(cls, work_dir: Path) -> set[str]:
        """获取工作目录的文件快照。

        Args:
            work_dir: 工作目录

        Returns:
            相对路径的集合
        """
        return {
            str(f.relative_to(work_dir)) for f in work_dir.rglob("*") if f.is_file()
        }


# 不对外暴露，仅供内部使用
__all__: list[str] = []
