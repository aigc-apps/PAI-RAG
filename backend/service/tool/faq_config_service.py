"""FAQ Config Service layer for database operations."""

from datetime import datetime, timezone
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from common.knowledgebase.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_FAQ_SIMILARITY_THRESHOLD
from loguru import logger

from db.models.faq_config import FAQConfigCreate
from db.models.chatbot import ChatBotEntity


def get_default_faq_config() -> dict:
    """Get default FAQ config values."""
    return {
        "active": True,
        "similarity_threshold": DEFAULT_FAQ_SIMILARITY_THRESHOLD,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "enable_question_in_retrieval": True,
        "enable_question_in_response": True,
        "enable_answer_in_retrieval": False,
        "enable_answer_in_response": True,
        "return_direct": False,
        "kb_id": None,
    }




class FAQConfigService:
    """Service layer for FAQ Config operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize FAQConfigService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_or_create_faq_config(
        self, chatbot: ChatBotEntity
    ) -> FAQConfigCreate:
        """
        Get or create a FAQ config for a chatbot.

        Args:
            chatbot: Chatbot entity

        Returns:
            FAQConfigCreate representing the FAQ config
        """
        # If faq_config exists and is not empty, return it
        if chatbot.faq_config:
            logger.info(
                f"Found existing FAQ config for chatbot_id: {chatbot.id}"
            )
            return FAQConfigCreate.model_validate(chatbot.faq_config)

        # Create new FAQ config with default values
        default_config = get_default_faq_config()
        chatbot.faq_config = default_config
        self.session.add(chatbot)

        await self.session.flush()
        await self.session.refresh(chatbot)
        logger.info(
            f"Created FAQ config to: {default_config}"
        )
        return FAQConfigCreate.model_validate(default_config)

    async def update_faq_config(
        self, chatbot: ChatBotEntity, update_data: FAQConfigCreate
    ) -> FAQConfigCreate:
        """
        Update FAQ config for a chatbot.
        Note: Caller is responsible for committing the session.

        Args:
            chatbot: Chatbot Entity
            update_data: Updated FAQ Config data

        Returns:
            Updated FAQConfigCreate

        Raises:
            ValueError: If Chatbot not found
        """

        logger.info(f"Updating FAQ Config for chatbot {chatbot.id} with data: {update_data}")

        # Get current config or use defaults
        current_config = chatbot.faq_config.copy() if chatbot.faq_config else get_default_faq_config()

        # Update fields from update_data
        update_dict = update_data.model_dump(exclude_unset=True)
        current_config.update(update_dict)

        # Update chatbot's faq_config
        chatbot.faq_config = current_config
        chatbot.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        flag_modified(chatbot, "faq_config")
        self.session.add(chatbot)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(chatbot)

        logger.info(f"Updated FAQ Config for chatbot: {chatbot.id}")
        return FAQConfigCreate.model_validate(chatbot.faq_config)
