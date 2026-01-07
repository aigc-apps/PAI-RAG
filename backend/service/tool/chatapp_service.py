"""ChatApp Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.chatbot import ChatBotCreate, ChatBotEntity
from db.models.knowledgebase.knowledgebase import KnowledgebaseCreate, RetrievalConfig, ChunkConfig, TableParserConfig
from common.chat.response_model import PagedResult
from common.knowledgebase.constants import FAQ_KNOWLEDGEBASE_NAME
from common.knowledgebase.types import VectorIndexRetrievalType
from service.knowledgebase.knowledgebase_service import KnowledgebaseService
from service.model.embedding_service import EmbeddingService
from service.tool.faq_config_service import FAQConfigService


class ChatappService:
    """Service layer for ChatApp (ChatBot) entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize ChatappService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def _ensure_faq_knowledgebase(self, chatbot_id: str, app_id: str, tenant_id: str) -> None:
        """
        Ensure FAQ knowledgebase exists for the given chatbot_id and app_id.
        Creates it if it doesn't exist.
        Uses embedding_model from faq_config if available, otherwise uses default.

        Args:
            chatbot_id: ChatApp chatbot_id
            app_id: ChatApp app_id
            tenant_id: Tenant ID
        """
        kb_name = f"{app_id}_{FAQ_KNOWLEDGEBASE_NAME}"
        knowledgebase_service = KnowledgebaseService(self.session)
        embedding_service = EmbeddingService(self.session)
        faq_config_service = FAQConfigService(self.session)

        knowledgebase = await knowledgebase_service.get_knowledgebase_by_name(kb_name, tenant_id=tenant_id)

        if not knowledgebase:
            logger.info(f"Creating FAQ knowledgebase {kb_name} for app_id {app_id} and tenant {tenant_id}")

            # Get FAQ config to get embedding_model
            faq_config = await faq_config_service.get_faq_config_by_chatbot_id(
                chatbot_id=chatbot_id, tenant_id=tenant_id
            )

            # Use embedding_model from faq_config if available, otherwise use default
            if faq_config and faq_config.embedding_model:
                embedding_model = faq_config.embedding_model
                logger.info(f"Using embedding_model {embedding_model} from FAQ config for knowledgebase {kb_name}")
            else:
                default_embedding_config = await embedding_service.get_default_embedding(tenant_id=tenant_id)
                embedding_model = default_embedding_config.model_id
                logger.info(f"Using default embedding_model {embedding_model} for knowledgebase {kb_name}")

            # Set default retrieval_config
            default_similarity_threshold = faq_config.similarity_threshold if faq_config else 0.9

            retrieval_config = RetrievalConfig(
                retrieval_mode=VectorIndexRetrievalType.vector,
                top_k=1,
                enable_rerank=False,
                rerank_top_k=None,
                vector_weight=1.0,
                similarity_threshold=default_similarity_threshold,
            )

            chunk_config = ChunkConfig(
                table_config=TableParserConfig(
                header_index_max=0,
                question_column_index=0,
                answer_column_index=1,
                ),
                parser_type="faq",
            )

            kb_create = KnowledgebaseCreate(
                name=kb_name,
                description="faq知识库",
                embedding_model=embedding_model,
                retrieval_config=retrieval_config,
                chunk_config=chunk_config,
            )
            knowledgebase = await knowledgebase_service.create_knowledgebase(kb_data=kb_create, tenant_id=tenant_id)
            await self.session.flush()
            await self.session.refresh(knowledgebase)
            logger.info(f"Created FAQ knowledgebase {knowledgebase.id} (name: {kb_name}) for app_id {app_id}")

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

            # If enable_faq is True, create FAQ config
            if app_data.enable_faq:
                # Initialize FAQ config with default values
                faq_config_service = FAQConfigService(self.session)
                await faq_config_service.get_or_create_faq_config(
                    chatbot_id=chatbot.id, tenant_id=tenant_id
                )

                # Ensure FAQ knowledgebase exists (uses embedding_model from faq_config)
                await self._ensure_faq_knowledgebase(chatbot.id, chatbot.app_id, tenant_id)

                await self.session.flush()
                await self.session.refresh(chatbot)

                logger.info(
                    f"Created FAQ config for ChatApp: {chatbot.id} (app_id: {chatbot.app_id})"
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

        if update_data.enable_faq:
            # Enable FAQ: create FAQ config if not exists
            if not chatbot.faq_config:
                faq_config_service = FAQConfigService(self.session)
                await faq_config_service.get_or_create_faq_config(
                    chatbot_id=chatbot.id, tenant_id=tenant_id
                )

                # Ensure FAQ knowledgebase exists (uses embedding_model from faq_config)
                await self._ensure_faq_knowledgebase(chatbot.id, chatbot.app_id, tenant_id)

                logger.info(
                    f"Created FAQ config for ChatApp: {chatbot.id}"
                )
        else:
            # Disable FAQ: clear faq_config (but keep FAQ items)
            chatbot.faq_config = None
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
        if update_data.prompts is not None:
            chatbot.prompts = update_data.prompts
        if update_data.enable_faq is not None:
            chatbot.enable_faq = update_data.enable_faq

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
