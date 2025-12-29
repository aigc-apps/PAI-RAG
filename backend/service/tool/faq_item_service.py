"""FAQ Item Service layer for database operations."""

from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger

from db.models.faq_item import FAQItemCreate, FAQItemEntity
from common.chat.response_model import PagedResult


class FAQItemService:
    """Service layer for FAQ Item entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize FAQItemService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

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
        faq_id: Optional[str] = None,
        tenant_id: str = None,
        page: int = 1,
        size: int = 100,
    ) -> PagedResult[List[FAQItemEntity]]:
        """
        List FAQ Item entities with pagination.

        Args:
            chatbot_id: Chatbot ID
            faq_id: Optional FAQ Config ID filter
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

        if faq_id is not None:
            base_query = base_query.where(FAQItemEntity.faq_id == faq_id)

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = (
            base_query.offset(offset).limit(size).order_by(FAQItemEntity.created_at.desc())
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
        faq_id: str,
        faq_item_data: FAQItemCreate,
        tenant_id: str,
    ) -> FAQItemEntity:
        """
        Create a new FAQ Item entity.
        Note: Caller is responsible for committing the session.

        Args:
            chatbot_id: Chatbot ID
            faq_id: FAQ Config ID
            faq_item_data: FAQ Item creation data
            tenant_id: Tenant ID

        Returns:
            Created FAQItemEntity (not yet committed)
        """
        faq_item = FAQItemEntity.model_validate(
            faq_item_data,
            update={"chatbot_id": chatbot_id, "faq_id": faq_id, "tenant_id": tenant_id},
        )
        self.session.add(faq_item)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(faq_item)

            logger.info(
                f"Created FAQ Item entity: {faq_item.id} (chatbot_id: {chatbot_id}, faq_id: {faq_id})"
            )
            return faq_item
        except Exception as e:
            logger.error(f"Error creating FAQ Item: {e}")
            raise ValueError(f"创建FAQ条目失败: {e}") from e

    async def update_faq_item(
        self, id: str, update_data: FAQItemCreate, tenant_id: str
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
            raise ValueError(f"FAQ条目 '{id}' 不存在。")

        logger.info(f"Updating FAQ Item {id} with data: {update_data}")

        # Update fields
        if update_data.question is not None:
            faq_item.question = update_data.question
        if update_data.answer is not None:
            faq_item.answer = update_data.answer
        if update_data.faq_id is not None:
            faq_item.faq_id = update_data.faq_id
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
        return faq_item

    async def delete_faq_item(self, id: str, tenant_id: str) -> None:
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
            raise ValueError(f"FAQ条目 '{id}' 不存在。")

        # Delete from database (staged, not committed)
        await self.session.delete(faq_item)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted FAQ Item entity: {id}")
