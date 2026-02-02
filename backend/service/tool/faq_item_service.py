"""FAQ Item Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger

from db.models.faq_item import FAQItemCreate, FAQItemEntity
from db.models.faq_config import FAQConfigCreate
from db.models.chatbot import ChatBotEntity
from common.chat.response_model import PagedResult
from service.knowledgebase.knowledgebase_service import KnowledgebaseService
from service.knowledgebase.rag_service import RagService
from pairag.file.utils.tokenization import estimate_tokens_in_text
from llama_index.core.schema import TextNode


class FAQItemService:
    """Service layer for FAQ Item entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize FAQItemService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_faq_knowledgebase(self, chatbot_id: str, tenant_id: str):
        """
        Get FAQ knowledgebase for the given chatbot.

        Args:
            chatbot_id: Chatbot ID
            tenant_id: Tenant ID

        Returns:
            Knowledgebase entity if found, None otherwise
        """
        # Get chatbot to get app_id
        chatbot = await self.session.exec(
            select(ChatBotEntity).where(
                ChatBotEntity.app_id == chatbot_id, ChatBotEntity.tenant_id == tenant_id
            )
        )
        chatbot = chatbot.first()
        if not chatbot:
            return None

        # Convert dict to FAQConfigCreate object
        if not chatbot.faq_config:
            return None

        try:
            faq_config = FAQConfigCreate.model_validate(chatbot.faq_config)
        except Exception as e:
            logger.warning(f"Failed to validate FAQ config for chatbot {chatbot_id}: {e}")
            return None

        if not faq_config.kb_id:
            return None

        knowledgebase_service = KnowledgebaseService(self.session)
        return await knowledgebase_service.get_knowledgebase(faq_config.kb_id, tenant_id=tenant_id)

    async def save_faq_to_knowledgebase(
        self, faq_item: FAQItemEntity, tenant_id: str, rag_service: RagService
    ) -> None:
        """
        Save FAQ item to knowledgebase.

        Args:
            faq_item: FAQ Item entity
            tenant_id: Tenant ID
        """
        try:
            if not faq_item.question or not faq_item.answer:
                logger.warning(
                    f"FAQ item {faq_item.id} has no question or answer, skipping save to KB"
                )
                return

            # Get FAQ knowledgebase
            kb = await self.get_faq_knowledgebase(faq_item.chatbot_id, tenant_id)
            if not kb:
                logger.warning(
                    f"FAQ knowledgebase not found for chatbot {faq_item.chatbot_id}, skipping save to KB"
                )
                return

            # Get FAQ config from chatbot to determine what to include in chunk_text
            chatbot = await self.session.exec(
                select(ChatBotEntity).where(
                    ChatBotEntity.app_id == faq_item.chatbot_id,
                    ChatBotEntity.tenant_id == tenant_id,
                )
            )
            chatbot = chatbot.first()

            faq_config = None
            if chatbot and chatbot.faq_config:
                faq_config = FAQConfigCreate.model_validate(chatbot.faq_config)

            # Build chunk_text based on faq_config settings
            chunk_parts = []
            if faq_config:
                if faq_config.enable_question_in_retrieval:
                    chunk_parts.append(f"{faq_item.question}")
                if faq_config.enable_answer_in_retrieval:
                    chunk_parts.append(f"{faq_item.answer}")
            else:
                chunk_parts.append(f"{faq_item.question}")

            chunk_text = "\n".join(chunk_parts) if chunk_parts else ""

            # Create metadata for TextNode
            node_metadata = {
                "faq_item_id": faq_item.id,
                "chatbot_id": faq_item.chatbot_id,
                "question": faq_item.question,
                "answer": faq_item.answer,
                "token_count": estimate_tokens_in_text(chunk_text),
            }

            # Create TextNode directly (no KbChunkEntity needed for FAQ items)
            kb_node = TextNode(
                id_=faq_item.id,
                text=chunk_text,
                metadata=node_metadata,
            )


            if faq_item.active:
                await rag_service.ainsert(kb_id=kb.id, nodes=[kb_node], tenant_id=tenant_id)
                logger.info(
                    f"Inserted FAQ item {faq_item.id} into knowledgebase {kb.id}"
                )
            else:
                logger.info(
                    f"FAQ item {faq_item.id} is inactive, skipping vector store insertion"
                )

        except Exception as e:
            logger.error(f"Failed to save FAQ item to knowledgebase: {e}")

    async def delete_faq_from_knowledgebase(
        self, faq_item: FAQItemEntity, tenant_id: str, rag_service: RagService
    ) -> None:
        """
        Delete FAQ item from knowledgebase.

        Args:
            faq_item: FAQ Item entity
            tenant_id: Tenant ID
        """
        try:
            # Get FAQ knowledgebase
            kb = await self.get_faq_knowledgebase(faq_item.chatbot_id, tenant_id)
            if not kb:
                logger.warning(
                    f"FAQ knowledgebase not found for chatbot {faq_item.chatbot_id}, skipping delete from KB"
                )
                return

            # Delete from vector store using faq_item.id as node_id
            await rag_service.adelete(kb_id=kb.id, node_ids=[faq_item.id], tenant_id=tenant_id)
            logger.info(
                f"Deleted FAQ item {faq_item.id} from knowledgebase {kb.id}"
            )

        except Exception as e:
            logger.error(f"Failed to delete FAQ item from knowledgebase: {e}")
            # Don't raise exception, just log the error

    async def get_faq_item(self, id: str, tenant_id: str) -> Optional[FAQItemEntity]:
        """
        Get a single FAQ Item entity by ID.

        Args:
            id: FAQ Item entity ID
            tenant_id: Tenant ID

        Returns:
            FAQItemEntity if found, None otherwise
        """
        faq_items = await self.session.exec(
            select(FAQItemEntity).where(
                FAQItemEntity.id == id, FAQItemEntity.tenant_id == tenant_id
            )
        )
        return faq_items.first()

    async def list_faq_items(
        self,
        chatbot_id: str,
        tenant_id: str = None,
        page: int = 1,
        size: int = 100,
    ) -> PagedResult[List[FAQItemEntity]]:
        """
        List FAQ Item entities with pagination.

        Args:
            chatbot_id: Chatbot ID
            tenant_id: Tenant ID
            page: Page number (1-indexed)
            size: Page size

        Returns:
            PagedResult containing list of FAQItemEntity and pagination metadata
        """
        # Build base query
        base_query = select(FAQItemEntity).where(
            FAQItemEntity.chatbot_id == chatbot_id,
            FAQItemEntity.tenant_id == tenant_id,
        )

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = (
            base_query.offset(offset).limit(size).order_by(FAQItemEntity.created_at.desc(), FAQItemEntity.id.asc())
        )
        results = await self.session.exec(paginated_query)
        faq_items = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=faq_items,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_faq_item(
        self,
        chatbot_id: str,
        faq_item_data: FAQItemCreate,
        tenant_id: str,
    ) -> FAQItemEntity:
        """
        Create a new FAQ Item entity.
        Note: Caller is responsible for committing the session.

        Args:
            chatbot_id: Chatbot ID
            faq_item_data: FAQ Item creation data
            tenant_id: Tenant ID

        Returns:
            Created FAQItemEntity (not yet committed)
        """
        faq_item = FAQItemEntity.model_validate(
            faq_item_data,
            update={"chatbot_id": chatbot_id, "tenant_id": tenant_id},
        )
        self.session.add(faq_item)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(faq_item)

            logger.info(
                f"Created FAQ Item entity: {faq_item.id} (chatbot_id: {chatbot_id})"
            )


            return faq_item
        except Exception as e:
            logger.error(f"Error creating FAQ Item: {e}")
            raise ValueError(f"Creating FAQ item failed: {e}") from e

    async def update_faq_item(
        self, id: str, update_data: FAQItemCreate, tenant_id: str, rag_service: Optional[RagService] = None
    ) -> FAQItemEntity:
        """
        Update an existing FAQ Item entity.
        Note: Caller is responsible for committing the session.

        Args:
            id: FAQ Item entity ID
            update_data: Updated FAQ Item data
            tenant_id: Tenant ID

        Returns:
            Updated FAQItemEntity (not yet committed)

        Raises:
            ValueError: If FAQ Item entity not found
        """
        faq_item = await self.get_faq_item(id=id, tenant_id=tenant_id)
        if not faq_item:
            raise ValueError(f"FAQ item '{id}' does not exist.")

        logger.info(f"Updating FAQ Item {id} with data: {update_data}")

        # Update fields
        if update_data.question is not None:
            faq_item.question = update_data.question
        if update_data.answer is not None:
            faq_item.answer = update_data.answer
        if update_data.chatbot_id is not None:
            faq_item.chatbot_id = update_data.chatbot_id
        if update_data.file_id is not None:
            faq_item.file_id = update_data.file_id
        if update_data.active is not None:
            faq_item.active = update_data.active

        faq_item.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.add(faq_item)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(faq_item)

        logger.info(f"Updated FAQ Item entity: {faq_item.id}")

        # Update FAQ in knowledgebase (delete old, insert new if active)
        await self.delete_faq_from_knowledgebase(faq_item, tenant_id, rag_service)
        await self.save_faq_to_knowledgebase(faq_item, tenant_id, rag_service)

        return faq_item

    async def delete_faq_item(self, id: str, tenant_id: str, rag_service: Optional[RagService] = None) -> None:
        """
        Delete a FAQ Item entity.
        Note: Caller is responsible for committing the session.

        Args:
            id: FAQ Item entity ID
            tenant_id: Tenant ID

        Raises:
            ValueError: If FAQ Item entity not found
        """
        faq_item = await self.get_faq_item(id=id, tenant_id=tenant_id)
        if not faq_item:
            raise ValueError(f"FAQ item '{id}' does not exist.")

        # Delete from knowledgebase first
        await self.delete_faq_from_knowledgebase(faq_item, tenant_id, rag_service)

        # Delete from database (staged, not committed)
        await self.session.delete(faq_item)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted FAQ Item entity: {id}")
