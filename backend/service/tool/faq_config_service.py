"""FAQ Config Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from common.knowledgebase.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_FAQ_SIMILARITY_THRESHOLD, FAQ_KNOWLEDGEBASE_NAME
from common.knowledgebase.types import VectorIndexRetrievalType
from loguru import logger

from db.models.faq_config import FAQConfigCreate
from db.models.chatbot import ChatBotEntity
from db.models.knowledgebase.knowledgebase import KnowledgebaseCreate, RetrievalConfig
from service.knowledgebase.knowledgebase_service import KnowledgebaseService


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
            "enable_question_in_response": True,
            "enable_answer_in_retrieval": False,
            "enable_answer_in_response": True,
            "return_direct": False,
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

    async def update_faq_config_with_sync(
        self,
        app_id: str,
        chatbot_id: str,
        update_data: FAQConfigCreate,
        tenant_id: str,
        knowledgebase_service: Optional[KnowledgebaseService] = None
    ) -> FAQConfigCreate:
        """
        Update FAQ config with full synchronization logic:
        - Get or create FAQ config
        - Sync chatbot.enable_faq with faq_config.active
        - Update FAQ config
        - Update corresponding knowledgebase if embedding_model or similarity_threshold changed

        Note: Caller is responsible for committing the session.

        Args:
            app_id: Chatbot app_id (used for knowledgebase name)
            chatbot_id: Chatbot ID
            update_data: Updated FAQ Config data
            tenant_id: Tenant ID
            knowledgebase_service: Optional KnowledgebaseService for updating knowledgebase

        Returns:
            Updated FAQConfigCreate

        Raises:
            ValueError: If Chatbot not found
        """
        # Get chatbot entity
        chatbot = await self.session.exec(
            select(ChatBotEntity).where(
                ChatBotEntity.id == chatbot_id,
                ChatBotEntity.tenant_id == tenant_id,
            )
        )
        chatbot = chatbot.first()

        if not chatbot:
            raise ValueError(f"Chatbot '{chatbot_id}' 不存在。")

        # Get or create FAQ config (this will use the same chatbot entity if it exists)
        await self.get_or_create_faq_config(
            chatbot_id=chatbot_id, tenant_id=tenant_id
        )

        # Refresh chatbot to get latest state
        await self.session.refresh(chatbot)

        # Sync chatbot.enable_faq with faq_config.active
        if update_data.active is not None:
            if chatbot.enable_faq != update_data.active:
                chatbot.enable_faq = update_data.active
                logger.info(f"Synced chatbot.enable_faq to {update_data.active} for chatbot {chatbot_id}")

        # Update FAQ config
        updated_faq_config = await self.update_faq_config(
            chatbot_id=chatbot_id,
            update_data=update_data,
            tenant_id=tenant_id
        )

        # Update corresponding knowledgebase if embedding_model or similarity_threshold changed
        if knowledgebase_service and (update_data.embedding_model is not None or update_data.similarity_threshold is not None):
            kb_name = f"{app_id}_{FAQ_KNOWLEDGEBASE_NAME}"
            kb = await knowledgebase_service.get_knowledgebase_by_name(kb_name, tenant_id=tenant_id)

            if kb:
                # Prepare update data for knowledgebase
                kb_update_data = KnowledgebaseCreate()
                update_fields = []

                # Update embedding_model if provided
                if update_data.embedding_model is not None:
                    kb_update_data.embedding_model = update_data.embedding_model
                    update_fields.append(f"embedding_model={update_data.embedding_model}")

                # Update retrieval_config.similarity_threshold if provided
                if update_data.similarity_threshold is not None:
                    # Get current retrieval_config or create default
                    current_retrieval_config = RetrievalConfig.model_validate(kb.retrieval_config) if kb.retrieval_config else RetrievalConfig(
                        retrieval_mode=VectorIndexRetrievalType.vector,
                        top_k=1,
                        enable_rerank=False,
                        rerank_top_k=None,
                        vector_weight=1.0,
                        similarity_threshold=update_data.similarity_threshold,
                    )
                    # Update similarity_threshold
                    current_retrieval_config.similarity_threshold = update_data.similarity_threshold
                    kb_update_data.retrieval_config = current_retrieval_config
                    update_fields.append(f"similarity_threshold={update_data.similarity_threshold}")

                # Update knowledgebase only if there are fields to update
                if kb_update_data.embedding_model is not None or kb_update_data.retrieval_config is not None:
                    await knowledgebase_service.update_knowledgebase(
                        kb_id=kb.id,
                        update_data=kb_update_data,
                        tenant_id=tenant_id
                    )
                    logger.info(f"Updated FAQ knowledgebase {kb_name} with {', '.join(update_fields)}")

        return updated_faq_config
