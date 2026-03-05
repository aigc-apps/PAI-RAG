"""Skill Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger
from db.models.skill import SkillCreate, SkillUpdate, SkillEntity
from common.chat.response_model import PagedResult


class SkillService:
    """Service layer for Skill entity CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_skill(self, skill_id: str, tenant_id: str) -> Optional[SkillEntity]:
        result = await self.session.exec(
            select(SkillEntity).where(
                SkillEntity.id == skill_id,
                SkillEntity.tenant_id == tenant_id,
            )
        )
        return result.first()

    async def get_skill_by_name(self, name: str, tenant_id: str) -> Optional[SkillEntity]:
        result = await self.session.exec(
            select(SkillEntity).where(
                SkillEntity.name == name,
                SkillEntity.tenant_id == tenant_id,
            )
        )
        return result.first()

    async def list_skills(
        self,
        tenant_id: str,
        page: int = 1,
        size: int = 50,
        enabled_only: bool = False,
    ) -> PagedResult[List[SkillEntity]]:
        base_query = select(SkillEntity).where(SkillEntity.tenant_id == tenant_id)
        if enabled_only:
            base_query = base_query.where(SkillEntity.enabled == True)

        # total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # paginated results
        offset = (page - 1) * size
        paginated_query = base_query.offset(offset).limit(size)
        results = await self.session.exec(paginated_query)
        skills = list(results.all())

        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=skills,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def get_enabled_skills(self, tenant_id: str) -> List[SkillEntity]:
        """Get all enabled skills for a tenant (no pagination, for agent use)."""
        result = await self.session.exec(
            select(SkillEntity).where(
                SkillEntity.tenant_id == tenant_id,
                SkillEntity.enabled == True,
            )
        )
        return list(result.all())

    async def create_skill(self, skill_data: SkillCreate, tenant_id: str) -> SkillEntity:
        skill = SkillEntity.model_validate(
            skill_data, update={"tenant_id": tenant_id}
        )
        self.session.add(skill)

        try:
            await self.session.flush()
            await self.session.refresh(skill)
            logger.info(f"Created skill: {skill.id} (name: {skill.name})")
            return skill
        except IntegrityError as e:
            logger.error(f"IntegrityError when creating skill: {e.orig}")
            raise ValueError(f"Skill creation failed: {e}") from e

    async def update_skill(
        self, skill_id: str, update_data: SkillUpdate, tenant_id: str
    ) -> SkillEntity:
        result = await self.session.exec(
            select(SkillEntity).where(
                SkillEntity.id == skill_id,
                SkillEntity.tenant_id == tenant_id,
            )
        )
        skill = result.first()
        if not skill:
            raise ValueError(f"Skill '{skill_id}' does not exist.")

        # Update non-None fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            if value is not None:
                setattr(skill, field, value)

        self.session.add(skill)
        await self.session.flush()
        await self.session.refresh(skill)

        logger.info(f"Updated skill: {skill.id} (name: {skill.name})")
        return skill

    async def toggle_skill(self, skill_id: str, enabled: bool, tenant_id: str) -> SkillEntity:
        """Toggle skill enabled status."""
        result = await self.session.exec(
            select(SkillEntity).where(
                SkillEntity.id == skill_id,
                SkillEntity.tenant_id == tenant_id,
            )
        )
        skill = result.first()
        if not skill:
            raise ValueError(f"Skill '{skill_id}' does not exist.")

        skill.enabled = enabled
        self.session.add(skill)
        await self.session.flush()
        await self.session.refresh(skill)

        logger.info(f"Toggled skill {skill.id}: enabled={enabled}")
        return skill

    async def delete_skill(self, skill_id: str, tenant_id: str) -> None:
        result = await self.session.exec(
            select(SkillEntity).where(
                SkillEntity.id == skill_id,
                SkillEntity.tenant_id == tenant_id,
            )
        )
        skill = result.first()
        if not skill:
            raise ValueError(f"Skill '{skill_id}' does not exist.")

        await self.session.delete(skill)
        await self.session.flush()

        logger.info(f"Deleted skill: {skill_id} (name: {skill.name})")
