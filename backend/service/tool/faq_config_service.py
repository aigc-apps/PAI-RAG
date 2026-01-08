"""FAQ Config Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from common.knowledgebase.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_FAQ_SIMILARITY_THRESHOLD
from loguru import logger

from db.models.faq_config import FAQConfigCreate
from db.models.chatbot import ChatBotEntity


class FAQConfigService:
    """Service layer for FAQ Config operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize FAQConfigService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    def _get_default_faq_config(self) -> dict:
        """Get default FAQ config values."""
        return {
            "active": True,
            "similarity_threshold": DEFAULT_FAQ_SIMILARITY_THRESHOLD,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "enable_question_in_retrieval": True,
            "enable_question_in_response": False,
            "enable_answer_in_retrieval": False,
            "enable_answer_in_response": True,
            "kb_id": None,
        }

    async def get_faq_config_by_chatbot_id(
        self, chatbot_id: str, tenant_id: str
    ) -> Optional[FAQConfigCreate]:
        """
        Get FAQ Config by chatbot_id.

        Args:
            chatbot_id: Chatbot ID
            tenant_id: Tenant ID

        Returns:
            FAQConfigCreate if found, None otherwise
        """
        chatbot = await self.session.exec(
            select(ChatBotEntity).where(
                ChatBotEntity.id == chatbot_id,
                ChatBotEntity.tenant_id == tenant_id,
            )
        )
        chatbot = chatbot.first()
        if not chatbot or not chatbot.faq_config:
            return None

        # Convert dict to FAQConfigCreate
        return FAQConfigCreate.model_validate(chatbot.faq_config)

    async def get_or_create_faq_config(
        self, chatbot_id: str, tenant_id: str
    ) -> FAQConfigCreate:
        """
        Get or create a FAQ config for a chatbot.

        Args:
            chatbot_id: Chatbot ID
            tenant_id: Tenant ID

        Returns:
            FAQConfigCreate representing the FAQ config
        """
        chatbot = await self.session.exec(
            select(ChatBotEntity).where(
                ChatBotEntity.id == chatbot_id,
                ChatBotEntity.tenant_id == tenant_id,
            )
        )
        chatbot = chatbot.first()

        if not chatbot:
            raise ValueError(f"Chatbot '{chatbot_id}' 不存在。")

        # If faq_config exists and is not empty, return it
        if chatbot.faq_config:
            logger.info(
                f"Found existing FAQ config for chatbot_id: {chatbot_id}"
            )
            return FAQConfigCreate.model_validate(chatbot.faq_config)

        # Create new FAQ config with default values
        default_config = self._get_default_faq_config()
        chatbot.faq_config = default_config
        self.session.add(chatbot)

        await self.session.flush()
        await self.session.refresh(chatbot)
        logger.info(
            f"Created FAQ config for chatbot_id: {chatbot_id}"
        )
        return FAQConfigCreate.model_validate(default_config)

    async def update_faq_config(
        self, chatbot_id: str, update_data: FAQConfigCreate, tenant_id: str
    ) -> FAQConfigCreate:
        """
        Update FAQ config for a chatbot.
        Note: Caller is responsible for committing the session.

        Args:
            chatbot_id: Chatbot ID
            update_data: Updated FAQ Config data
            tenant_id: Tenant ID

        Returns:
            Updated FAQConfigCreate

        Raises:
            ValueError: If Chatbot not found
        """
        chatbot = await self.session.exec(
            select(ChatBotEntity).where(
                ChatBotEntity.id == chatbot_id,
                ChatBotEntity.tenant_id == tenant_id,
            )
        )
        chatbot = chatbot.first()

        if not chatbot:
            raise ValueError(f"Chatbot '{chatbot_id}' 不存在。")

        logger.info(f"Updating FAQ Config for chatbot {chatbot_id} with data: {update_data}")

        # Get current config or use defaults
        current_config = chatbot.faq_config.copy() if chatbot.faq_config else self._get_default_faq_config()

        # Update fields from update_data
        update_dict = update_data.model_dump(exclude_unset=True)
        current_config.update(update_dict)

        # Update chatbot's faq_config
        chatbot.faq_config = current_config
        chatbot.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.add(chatbot)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(chatbot)

        logger.info(f"Updated FAQ Config for chatbot: {chatbot_id}")
        return FAQConfigCreate.model_validate(chatbot.faq_config)
