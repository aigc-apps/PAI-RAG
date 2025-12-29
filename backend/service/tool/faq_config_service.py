"""FAQ Config Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger

from db.models.faq_config import FAQConfigCreate, FAQConfigEntity


class FAQConfigService:
    """Service layer for FAQ Config entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize FAQConfigService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_faq_config(self, id: str, tenant_id: str) -> Optional[FAQConfigEntity]:
        """
        Get a single FAQ Config entity by ID.

        Args:
            id: FAQ Config entity ID
            tenant_id: Tenant ID

        Returns:
            FAQConfigEntity if found, None otherwise
        """
        faq_configs = await self.session.exec(
            select(FAQConfigEntity).where(
                FAQConfigEntity.id == id, FAQConfigEntity.tenant_id == tenant_id
            )
        )
        return faq_configs.first()

    async def get_faq_config_by_chatbot_id(
        self, chatbot_id: str, tenant_id: str
    ) -> Optional[FAQConfigEntity]:
        """
        Get FAQ Config entity by chatbot_id.

        Args:
            chatbot_id: Chatbot ID
            tenant_id: Tenant ID

        Returns:
            FAQConfigEntity if found, None otherwise
        """
        faq_configs = await self.session.exec(
            select(FAQConfigEntity).where(
                FAQConfigEntity.chatbot_id == chatbot_id,
                FAQConfigEntity.tenant_id == tenant_id,
            )
        )
        return faq_configs.first()

    async def get_or_create_faq_config(
        self, chatbot_id: str, tenant_id: str
    ) -> FAQConfigEntity:
        """
        Get or create a FAQ config entity for a chatbot.

        Args:
            chatbot_id: Chatbot ID
            tenant_id: Tenant ID

        Returns:
            FAQConfigEntity representing the FAQ config (not yet committed if newly created)
        """
        # Try to find existing FAQ config
        faq_config = await self.get_faq_config_by_chatbot_id(
            chatbot_id=chatbot_id, tenant_id=tenant_id
        )

        if faq_config:
            logger.info(
                f"Found existing FAQ config: {faq_config.id} for chatbot_id: {chatbot_id}"
            )
            return faq_config

        # Create new FAQ config with default values
        faq_config = FAQConfigEntity(
            chatbot_id=chatbot_id,
            tenant_id=tenant_id,
            score_threshold=0.9,
            embedding_model="BAAI/bge-m3",
            question_in_retrieval=True,
            question_in_response=False,
            answer_in_retrieval=False,
            answer_in_response=True,
        )
        self.session.add(faq_config)

        try:
            await self.session.flush()
            await self.session.refresh(faq_config)
            logger.info(
                f"Created FAQ config: {faq_config.id} for chatbot_id: {chatbot_id}"
            )
            return faq_config
        except Exception as e:
            logger.error(f"Error creating FAQ config: {e}")
            raise ValueError(f"创建FAQ配置失败: {e}") from e

    async def update_faq_config(
        self, id: str, update_data: FAQConfigCreate, tenant_id: str
    ) -> FAQConfigEntity:
        """
        Update an existing FAQ Config entity.
        Note: Caller is responsible for committing the session.

        Args:
            id: FAQ Config entity ID
            update_data: Updated FAQ Config data
            tenant_id: Tenant ID

        Returns:
            Updated FAQConfigEntity (not yet committed)

        Raises:
            ValueError: If FAQ Config entity not found
        """
        faq_config = await self.get_faq_config(id=id, tenant_id=tenant_id)
        if not faq_config:
            raise ValueError(f"FAQ配置 '{id}' 不存在。")

        logger.info(f"Updating FAQ Config {id} with data: {update_data}")

        # Update active field
        if update_data.active is not None:
            faq_config.active = update_data.active

        # Update individual config fields directly
        if update_data.score_threshold is not None:
            faq_config.score_threshold = update_data.score_threshold
        if update_data.embedding_model is not None:
            faq_config.embedding_model = update_data.embedding_model
        if update_data.question_in_retrieval is not None:
            faq_config.question_in_retrieval = update_data.question_in_retrieval
        if update_data.question_in_response is not None:
            faq_config.question_in_response = update_data.question_in_response
        if update_data.answer_in_retrieval is not None:
            faq_config.answer_in_retrieval = update_data.answer_in_retrieval
        if update_data.answer_in_response is not None:
            faq_config.answer_in_response = update_data.answer_in_response

        faq_config.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.add(faq_config)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(faq_config)

        logger.info(f"Updated FAQ Config entity: {faq_config.id}")
        return faq_config
