"""ChatApp Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.chatbot import ChatBotCreate, ChatBotEntity
from db.models.faq_config import FAQConfigEntity
from common.chat.response_model import PagedResult


class ChatappService:
    """Service layer for ChatApp (ChatBot) entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize ChatappService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_chatapp(self, id: str, tenant_id: str) -> Optional[ChatBotEntity]:
        """
        Get a single ChatApp entity by ID.

        Args:
            id: ChatApp entity ID

        Returns:
            ChatBotEntity if found, None otherwise
        """
        chatapps = await self.session.exec(select(ChatBotEntity).where(ChatBotEntity.id == id, ChatBotEntity.tenant_id == tenant_id))
        return chatapps.first()

    async def get_chatapp_by_app_id(self, app_id: str, tenant_id: str) -> Optional[ChatBotEntity]:
        """
        Get a single ChatApp entity by app_id.

        Args:
            app_id: ChatApp app_id

        Returns:
            ChatBotEntity if found, None otherwise
        """
        statement = select(ChatBotEntity).where(ChatBotEntity.app_id == app_id, ChatBotEntity.tenant_id == tenant_id)
        chatapps = await self.session.exec(statement)
        return chatapps.first()

    async def list_chatapps(
        self,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
        app_id: Optional[str] = None,
    ) -> PagedResult[List[ChatBotEntity]]:
        """
        List ChatApp entities with pagination and optional filtering.

        Args:
            page: Page number (1-indexed)
            size: Page size
            app_id: Optional filter for app_id

        Returns:
            PagedResult containing list of ChatBotEntity and pagination metadata
        """
        # Build base query
        base_query = select(ChatBotEntity).where(ChatBotEntity.tenant_id == tenant_id)

        # Add app_id filter if provided
        if app_id is not None:
            base_query = base_query.where(ChatBotEntity.app_id == app_id)

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = base_query.offset(offset).limit(size)
        results = await self.session.exec(paginated_query)
        apps = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=apps,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_chatapp(self, app_data: ChatBotCreate, tenant_id: str) -> ChatBotEntity:
        """
        Create a new ChatApp entity.
        Note: Caller is responsible for committing the session.

        Args:
            app_data: ChatApp creation data
            tenant_id: Tenant ID

        Returns:
            Created ChatBotEntity (not yet committed)

        Raises:
            ValueError: If app_id already exists (IntegrityError converted)
        """
        chatbot = ChatBotEntity.model_validate(app_data, update={"tenant_id": tenant_id})
        self.session.add(chatbot)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(chatbot)

            # If enable_faq is True, create FAQ config and set faq_id
            if app_data.enable_faq:
                faq_config = FAQConfigEntity(
                    chatbot_id=chatbot.id,
                    tenant_id=tenant_id,
                    score_threshold=0.9,
                    embedding_model="BAAI/bge-m3",
                    question_in_retrieval=True,
                    question_in_response=False,
                    answer_in_retrieval=False,
                    answer_in_response=True,
                )
                self.session.add(faq_config)
                await self.session.flush()
                await self.session.refresh(faq_config)

                # Update chatbot with faq_id
                chatbot.faq_id = faq_config.id
                await self.session.flush()
                await self.session.refresh(chatbot)

                logger.info(
                    f"Created FAQ config: {faq_config.id} for ChatApp: {chatbot.id} (app_id: {chatbot.app_id})"
                )

            logger.info(
                f"Created ChatApp entity: {chatbot.id} (app_id: {chatbot.app_id})"
            )
            return chatbot

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating ChatApp: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"应用ID '{app_data.app_id}' 已经存在。"
                ) from e
            else:
                raise ValueError(f"应用创建失败: {e}") from e

    async def update_chatapp(
        self, id: str, update_data: ChatBotCreate, tenant_id: str
    ) -> ChatBotEntity:
        """
        Update an existing ChatApp entity.
        Note: Caller is responsible for committing the session.

        Args:
            id: ChatApp entity ID
            update_data: Updated ChatApp data

        Returns:
            Updated ChatBotEntity (not yet committed)

        Raises:
            ValueError: If ChatApp entity not found
        """
        chatbot = await self.get_chatapp(id=id, tenant_id=tenant_id)
        if not chatbot:
            raise ValueError(f"应用 '{id}' 不存在。")

        logger.info(f"Updating ChatApp {id} with data: {update_data}")

        # Handle enable_faq field: create FAQ config if enabling, clear faq_id if disabling
        if update_data.enable_faq is not None:
            if update_data.enable_faq:
                # Enable FAQ: create FAQ config if not exists
                if not chatbot.faq_id:
                    faq_config = FAQConfigEntity(
                        chatbot_id=chatbot.id,
                        tenant_id=tenant_id,
                        score_threshold=0.9,
                        embedding_model="BAAI/bge-m3",
                        question_in_retrieval=True,
                        question_in_response=False,
                        answer_in_retrieval=False,
                        answer_in_response=True,
                    )
                    self.session.add(faq_config)
                    await self.session.flush()
                    await self.session.refresh(faq_config)
                    chatbot.faq_id = faq_config.id
                    logger.info(
                        f"Created FAQ config: {faq_config.id} for ChatApp: {chatbot.id}"
                    )
            else:
                # Disable FAQ: clear faq_id (but keep FAQ config and items)
                chatbot.faq_id = None
                logger.info(f"Disabled FAQ for ChatApp: {chatbot.id}")

        # Update fields
        if update_data.app_id is not None:
            chatbot.app_id = update_data.app_id
        if update_data.model_id is not None:
            chatbot.model_id = update_data.model_id
        if update_data.enable_search is not None:
            chatbot.enable_search = update_data.enable_search
        if update_data.enable_agent is not None:
            chatbot.enable_agent = update_data.enable_agent
        if update_data.enable_chatdb is not None:
            chatbot.enable_chatdb = update_data.enable_chatdb
        if update_data.kb_ids is not None:
            chatbot.kb_ids = update_data.kb_ids
        if update_data.mcp_ids is not None:
            chatbot.mcp_ids = update_data.mcp_ids
        if update_data.description is not None:
            chatbot.description = update_data.description
        if update_data.enable_vision is not None:
            chatbot.enable_vision = update_data.enable_vision
        if update_data.enable_input_guardrail is not None:
            chatbot.enable_input_guardrail = update_data.enable_input_guardrail
        if update_data.enable_output_guardrail is not None:
            chatbot.enable_output_guardrail = update_data.enable_output_guardrail
        if update_data.guardrail_hint is not None:
            chatbot.guardrail_hint = update_data.guardrail_hint
        # Only update faq_id if enable_faq is not provided (to allow manual faq_id updates)
        if update_data.faq_id is not None and update_data.enable_faq is None:
            chatbot.faq_id = update_data.faq_id
        if update_data.prompts is not None:
            chatbot.prompts = update_data.prompts

        chatbot.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.add(chatbot)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(chatbot)

        logger.info(f"Updated ChatApp entity: {chatbot.id} (app_id: {chatbot.app_id})")
        return chatbot

    async def delete_chatapp(self, id: str, tenant_id: str) -> None:
        """
        Delete a ChatApp entity.
        Note: Caller is responsible for committing the session.

        Args:
            id: ChatApp entity ID

        Raises:
            ValueError: If ChatApp entity not found
        """
        chatbot = await self.get_chatapp(id=id, tenant_id=tenant_id)
        if not chatbot:
            raise ValueError(f"应用 '{id}' 不存在。")

        # Delete from database (staged, not committed)
        await self.session.delete(chatbot)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted ChatApp entity: {id} (app_id: {chatbot.app_id})")

    async def get_all_chatapps(self, tenant_id: str) -> List[ChatBotEntity]:
        """
        Get all ChatApp entities without pagination.

        Returns:
            List of all ChatBotEntity
        """
        statement = select(ChatBotEntity).where(ChatBotEntity.tenant_id == tenant_id)
        chatapps = await self.session.exec(statement)
        return list(chatapps.all())
