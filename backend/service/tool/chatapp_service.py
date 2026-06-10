"""ChatApp Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.chatbot import ChatBotCreate, ChatBotEntity
from db.models.knowledgebase.knowledgebase import KnowledgebaseCreate, RetrievalConfig, KbEntity
from pairag.file.nodeparsers.file_parser import ChunkConfig, TableParserConfig

from common.chat.response_model import PagedResult
from common.knowledgebase.constants import FAQ_KNOWLEDGEBASE_NAME, DEFAULT_FAQ_SIMILARITY_THRESHOLD
from common.knowledgebase.types import VectorIndexRetrievalType
from service.knowledgebase.knowledgebase_service import KnowledgebaseService
from service.model.embedding_service import EmbeddingService
from service.tool.faq_config_service import FAQConfigService
from db.models.faq_config import FAQConfigCreate
from db.models.faq_item import FAQItemEntity
from service.knowledgebase.rag_service import RagService


class ChatappService:
    """Service layer for ChatApp (ChatBot) entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize ChatappService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def _ensure_faq_knowledgebase(self, faq_config: FAQConfigCreate, app_id: str, tenant_id: str) -> KbEntity:
        """
        Ensure FAQ knowledgebase exists for the given chatbot_id and app_id.
        Creates it if it doesn't exist.
        Uses embedding_model from faq_config if available, otherwise uses default.

        Args:
            chatbot_id: ChatApp chatbot_id
            app_id: ChatApp app_id
            tenant_id: Tenant ID

        Returns:
            KbEntity representing the FAQ knowledgebase
        """
        kb_name = f"{app_id}_{FAQ_KNOWLEDGEBASE_NAME}"
        knowledgebase_service = KnowledgebaseService(self.session)
        embedding_service = EmbeddingService(self.session)
        knowledgebase = await knowledgebase_service.get_knowledgebase_by_name(kb_name, tenant_id=tenant_id)

        if not knowledgebase:
            logger.info(f"Creating FAQ knowledgebase {kb_name} for app_id {app_id} and tenant {tenant_id}")


            # Use embedding_model from faq_config if available, otherwise use default
            if faq_config and faq_config.embedding_model:
                embedding_model = faq_config.embedding_model
                logger.info(f"Using embedding_model {embedding_model} from FAQ config for knowledgebase {kb_name}")
            else:
                default_embedding_config = await embedding_service.get_default_embedding(tenant_id=tenant_id)
                embedding_model = default_embedding_config.model_id
                logger.info(f"Using default embedding_model {embedding_model} for knowledgebase {kb_name}")

            # Set default retrieval_config
            default_similarity_threshold = faq_config.similarity_threshold if faq_config else DEFAULT_FAQ_SIMILARITY_THRESHOLD

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
                description="faq knowledgebase",
                embedding_model=embedding_model,
                retrieval_config=retrieval_config,
                chunk_config=chunk_config,
            )
            knowledgebase = await knowledgebase_service.create_knowledgebase(kb_data=kb_create, tenant_id=tenant_id)
            await self.session.flush()
            await self.session.refresh(knowledgebase)
            logger.info(f"Created FAQ knowledgebase {knowledgebase.id} (name: {kb_name}) for app_id {app_id}")

        return knowledgebase

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
            ValueError: If app_id already exists
        """
        # Check if app_id already exists
        existing_chatbot = await self.get_chatapp_by_app_id(app_id=app_data.app_id, tenant_id=tenant_id)
        if existing_chatbot:
            raise ValueError(f"ChatApp ID '{app_data.app_id}' already exists.")

        chatbot = ChatBotEntity.model_validate(app_data, update={"tenant_id": tenant_id})
        self.session.add(chatbot)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(chatbot)

            # If enable_faq is True, create FAQ config
            if app_data.enable_faq:
                # Initialize FAQ config with default values and set kb_id
                faq_config_service = FAQConfigService(self.session)
                faq_config = await faq_config_service.get_or_create_faq_config(
                    chatbot=chatbot
                )

                # Ensure FAQ knowledgebase exists first (to get kb_id)
                knowledgebase = await self._ensure_faq_knowledgebase(faq_config=faq_config, app_id=chatbot.app_id, tenant_id=tenant_id)

                # Update faq_config with kb_id
                if not faq_config.kb_id:
                    faq_config.kb_id = knowledgebase.id
                    await faq_config_service.update_faq_config(
                        chatbot=chatbot,
                        update_data=faq_config
                    )

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
                    f"ChatApp ID '{app_data.app_id}' already exists."
                ) from e
            else:
                raise ValueError(f"ChatApp creation failed: {e}") from e

    async def update_chatapp(
        self, id: str, update_data: ChatBotCreate, tenant_id: str
    ) -> ChatBotEntity:
        """
        Update an existing ChatApp entity.
        Note: Caller is responsible for committing the session.

        Args:
            id: ChatApp entity ID
            update_data: Updated ChatApp data
            tenant_id: Tenant ID

        Returns:
            Updated ChatBotEntity (not yet committed)

        Raises:
            ValueError: If ChatApp entity not found or app_id already exists
        """
        chatbot = await self.get_chatapp(id=id, tenant_id=tenant_id)
        if not chatbot:
            raise ValueError(f"ChatApp '{id}' does not exist.")

        logger.info(f"Updating ChatApp {id} with data: {update_data}")

        # Check if app_id is being updated and if it conflicts with existing records
        if update_data.app_id is not None and update_data.app_id != chatbot.app_id:
            existing_chatbot = await self.get_chatapp_by_app_id(app_id=update_data.app_id, tenant_id=tenant_id)
            if existing_chatbot and existing_chatbot.id != id:
                raise ValueError(f"Application ID '{update_data.app_id}' already exists, update failed.")

        faq_config_service = FAQConfigService(self.session)
        if update_data.faq_config:
            current_config = FAQConfigCreate.model_validate(update_data.faq_config)
            await faq_config_service.update_faq_config(
                chatbot=chatbot,
                update_data=current_config
            )
            logger.info(f"Updated FAQ config to: {current_config}")

        # Update fields
        if update_data.app_id is not None:
            chatbot.app_id = update_data.app_id
        if update_data.model_id is not None:
            chatbot.model_id = update_data.model_id
        # Allow explicit null to clear the optional multimodal model selection.
        if "vision_model_id" in update_data.model_fields_set:
            chatbot.vision_model_id = update_data.vision_model_id
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
        if update_data.enable_auto_metadata_filter is not None:
            chatbot.enable_auto_metadata_filter = update_data.enable_auto_metadata_filter

        chatbot.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.add(chatbot)

        # If FAQ is enabled, ensure faq_config and FAQ knowledgebase exist
        if chatbot.enable_faq:
            if not chatbot.faq_config:
                faq_config = await faq_config_service.get_or_create_faq_config(chatbot=chatbot)
            else:
                faq_config = FAQConfigCreate.model_validate(chatbot.faq_config)


            knowledgebase = await self._ensure_faq_knowledgebase(
                faq_config=faq_config,
                app_id=chatbot.app_id,
                tenant_id=tenant_id
            )

            if not faq_config.kb_id:
                faq_config.kb_id = knowledgebase.id
                await faq_config_service.update_faq_config(
                    chatbot=chatbot,
                    update_data=faq_config
                )

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(chatbot)

        logger.info(f"Updated ChatApp entity: {chatbot.id} (app_id: {chatbot.app_id})")
        return chatbot

    async def delete_chatapp(self, id: str, tenant_id: str, rag_service: Optional[RagService] = None) -> None:
        """
        Delete a ChatApp entity.
        Note: Caller is responsible for committing the session.

        Args:
            id: ChatApp entity ID
            tenant_id: Tenant ID
            rag_service: Optional RagService for deleting FAQ knowledgebase

        Raises:
            ValueError: If ChatApp entity not found
        """
        chatbot = await self.get_chatapp(id=id, tenant_id=tenant_id)
        if not chatbot:
            raise ValueError(f"ChatApp '{id}' does not exist.")

        # Delete FAQ knowledgebase if exists
        if chatbot.faq_config:
            try:
                faq_config = FAQConfigCreate.model_validate(chatbot.faq_config)
                if faq_config.kb_id and rag_service:
                    try:
                        await rag_service.delete_knowledgebase(kb_id=faq_config.kb_id, tenant_id=tenant_id)
                        logger.info(f"Deleted FAQ knowledgebase {faq_config.kb_id} for ChatApp: {id}")
                    except Exception as e:
                        logger.warning(f"Failed to delete FAQ knowledgebase {faq_config.kb_id}: {e}")
                        # Continue with chatbot deletion even if KB deletion fails
            except Exception as e:
                logger.warning(f"Failed to parse FAQ config for chatbot {id}: {e}")
                # Continue with chatbot deletion even if FAQ config parsing fails

        try:
            faq_items = await self.session.exec(
                select(FAQItemEntity).where(
                    FAQItemEntity.chatbot_id == chatbot.app_id,
                    FAQItemEntity.tenant_id == tenant_id
                )
            )
            faq_items_list = list(faq_items.all())
            if faq_items_list:
                for faq_item in faq_items_list:
                    await self.session.delete(faq_item)
                logger.info(f"Deleted {len(faq_items_list)} FAQ items for ChatApp: {id}")
        except Exception as e:
            logger.warning(f"Failed to delete FAQ items for chatbot {id}: {e}")

        # Delete from database (staged, not committed)
        # FAQ items should be automatically deleted via CASCADE foreign key constraint if it exists
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
