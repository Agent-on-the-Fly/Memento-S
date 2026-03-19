"""Skill service for managing skills with vector storage."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from middleware.storage.models import Skill, SkillStatus
from middleware.storage.schemas import (
    SkillCreate,
    SkillRead,
    SkillUpdate,
    SkillWithEmbedding,
)
from middleware.storage.utils import get_east_8_time
from .base_service import BaseService


class SkillService(BaseService):
    """Service for managing skills with automatic database session management."""

    # Public API - auto-managed session

    async def create(self, data: SkillCreate) -> SkillRead:
        """Create a new skill.

        Args:
            data: Skill creation data

        Returns:
            Created skill
        """
        return await self._with_session(lambda db: self._create(db, data))

    async def get(self, skill_id: str) -> SkillRead | None:
        """Get skill by ID.

        Args:
            skill_id: Skill ID

        Returns:
            Skill or None if not found
        """
        return await self._with_session(lambda db: self._get(db, skill_id))

    async def get_by_name(self, name: str) -> SkillRead | None:
        """Get skill by name.

        Args:
            name: Skill unique name

        Returns:
            Skill or None if not found
        """
        return await self._with_session(lambda db: self._get_by_name(db, name))

    async def update(self, skill_id: str, data: SkillUpdate) -> SkillRead | None:
        """Update skill.

        Args:
            skill_id: Skill ID
            data: Update data

        Returns:
            Updated skill or None if not found
        """
        return await self._with_session(lambda db: self._update(db, skill_id, data))

    async def delete(self, skill_id: str) -> bool:
        """Delete skill.

        Args:
            skill_id: Skill ID

        Returns:
            True if deleted, False if not found
        """
        return await self._with_session(lambda db: self._delete(db, skill_id))

    async def list_active(self, limit: int = 1000) -> list[SkillRead]:
        """List all active skills.

        Args:
            limit: Maximum number of skills to return

        Returns:
            List of active skills ordered by name
        """
        return await self._with_session(lambda db: self._list_active(db, limit))

    async def list_by_category(self, category: str) -> list[SkillRead]:
        """Search skills by category.

        Args:
            category: Skill category

        Returns:
            List of skills in the category
        """
        return await self._with_session(lambda db: self._list_by_category(db, category))

    async def get_with_embedding(self, name: str) -> SkillWithEmbedding | None:
        """Get skill by name including embedding bytes.

        Args:
            name: Skill unique name

        Returns:
            SkillWithEmbedding or None if not found
        """
        return await self._with_session(lambda db: self._get_with_embedding(db, name))

    # Private implementation - requires manual session

    async def _create(self, db: AsyncSession, data: SkillCreate) -> SkillRead:
        """Create skill implementation."""
        obj = Skill(
            name=data.name,
            display_name=data.display_name,
            description=data.description,
            version=data.version,
            author=data.author,
            source_type=data.source_type,
            source_url=data.source_url,
            local_path=data.local_path,
            embedding=data.embedding,
            tags=data.tags,
            category=data.category,
            meta_info=data.meta_info,
            status=SkillStatus.ACTIVE.value,
            created_at=get_east_8_time(),
            updated_at=get_east_8_time(),
        )
        db.add(obj)
        await db.commit()
        await db.refresh(obj)
        return SkillRead.model_validate(obj)

    async def _get(self, db: AsyncSession, skill_id: str) -> SkillRead | None:
        """Get skill implementation."""
        obj = await db.get(Skill, skill_id)
        if obj is None:
            return None
        return SkillRead.model_validate(obj)

    async def _get_by_name(self, db: AsyncSession, name: str) -> SkillRead | None:
        """Get by name implementation."""
        stmt = select(Skill).where(Skill.name == name)
        result = await db.execute(stmt)
        obj = result.scalar_one_or_none()
        if obj is None:
            return None
        return SkillRead.model_validate(obj)

    async def _get_with_embedding(
        self, db: AsyncSession, name: str
    ) -> SkillWithEmbedding | None:
        """Get by name with embedding implementation."""
        stmt = select(Skill).where(Skill.name == name)
        result = await db.execute(stmt)
        obj = result.scalar_one_or_none()
        if obj is None:
            return None
        return SkillWithEmbedding.model_validate(obj)

    async def _update(
        self, db: AsyncSession, skill_id: str, data: SkillUpdate
    ) -> SkillRead | None:
        """Update skill implementation."""
        obj = await db.get(Skill, skill_id)
        if obj is None:
            return None

        if data.display_name is not None:
            obj.display_name = data.display_name
        if data.description is not None:
            obj.description = data.description
        if data.version is not None:
            obj.version = data.version
        if data.status is not None:
            obj.status = data.status
        if data.tags is not None:
            obj.tags = data.tags
        if data.category is not None:
            obj.category = data.category
        if data.meta_info is not None:
            obj.meta_info = data.meta_info

        # Update timestamp to East 8 time
        obj.updated_at = get_east_8_time()

        db.add(obj)
        await db.commit()
        await db.refresh(obj)
        return SkillRead.model_validate(obj)

    async def _delete(self, db: AsyncSession, skill_id: str) -> bool:
        """Delete skill implementation."""
        obj = await db.get(Skill, skill_id)
        if obj is None:
            return False
        await db.delete(obj)
        await db.commit()
        return True

    async def _list_active(self, db: AsyncSession, limit: int) -> list[SkillRead]:
        """List active skills implementation."""
        stmt = (
            select(Skill)
            .where(Skill.status == SkillStatus.ACTIVE.value)
            .order_by(Skill.name)
            .limit(limit)
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()
        return [SkillRead.model_validate(x) for x in rows]

    async def _list_by_category(
        self, db: AsyncSession, category: str
    ) -> list[SkillRead]:
        """List by category implementation."""
        stmt = (
            select(Skill)
            .where(Skill.category == category)
            .where(Skill.status == SkillStatus.ACTIVE.value)
            .order_by(Skill.name)
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()
        return [SkillRead.model_validate(x) for x in rows]

    async def list_all_names(self) -> set[str]:
        """获取DB中所有skill名称（包括所有状态）。

        Returns:
            Set of all skill names in database
        """
        return await self._with_session(lambda db: self._list_all_names(db))

    async def _list_all_names(self, db: AsyncSession) -> set[str]:
        """List all skill names implementation."""
        stmt = select(Skill.name)
        result = await db.execute(stmt)
        rows = result.scalars().all()
        return set(rows)
