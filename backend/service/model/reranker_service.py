"""Reranker Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.knowledgebase.reranker import (
    RerankerModelCreate,
    RerankerModelEntity,
)
from common.encrypt_utils import encrypt_key
from common.chat.response_model import PagedResult


class RerankerService:
    """Service layer for Reranker entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize RerankerService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_reranker(self, reranker_id: str) -> Optional[RerankerModelEntity]:
        """
        Get a single Reranker entity by ID.

        Args:
            reranker_id: Reranker entity ID

        Returns:
            RerankerModelEntity if found, None otherwise
        """
        return await self.session.get(RerankerModelEntity, reranker_id)

    async def get_reranker_by_model_id(
        self, model_id: str
    ) -> Optional[RerankerModelEntity]:
        """
        Get a single Reranker entity by model_id.

        Args:
            model_id: Reranker model_id

        Returns:
            RerankerModelEntity if found, None otherwise
        """
        statement = select(RerankerModelEntity).where(
            RerankerModelEntity.model_id == model_id
        )
        result = await self.session.exec(statement)
        return result.first()

    async def get_reranker_by_model_name(
        self, model_name: str
    ) -> Optional[RerankerModelEntity]:
        """
        Get a single Reranker entity by model_name.

        Args:
            model_name: Reranker model_name

        Returns:
            RerankerModelEntity if found, None otherwise
        """
        statement = select(RerankerModelEntity).where(
            RerankerModelEntity.model_name == model_name
        )
        result = await self.session.exec(statement)
        return result.first()

    async def list_rerankers(
        self,
        page: int = 1,
        size: int = 10,
        model_name: Optional[str] = None,
    ) -> PagedResult[List[RerankerModelEntity]]:
        """
        List Reranker entities with pagination and optional filtering.

        Args:
            page: Page number (1-indexed)
            size: Page size
            model_name: Optional filter for model_name

        Returns:
            PagedResult containing list of RerankerModelEntity and pagination metadata
        """
        # Build base query
        base_query = select(RerankerModelEntity)

        # Add model_name filter if provided
        if model_name is not None:
            base_query = base_query.where(
                RerankerModelEntity.model_name == model_name
            )

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = base_query.offset(offset).limit(size)
        results = await self.session.exec(paginated_query)
        rerankers = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=rerankers,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_reranker(
        self, reranker_data: RerankerModelCreate
    ) -> RerankerModelEntity:
        """
        Create a new Reranker entity.
        Note: Caller is responsible for committing the session.

        Args:
            reranker_data: Reranker creation data

        Returns:
            Created RerankerModelEntity (not yet committed)

        Raises:
            ValueError: If model_id already exists (IntegrityError converted)
        """
        # Encrypt API key
        encrypted_api_key = (
            encrypt_key(reranker_data.api_key) if reranker_data.api_key else None
        )

        # Create entity
        reranker = RerankerModelEntity.model_validate(
            reranker_data, update={"encrypted_api_key": encrypted_api_key}
        )

        self.session.add(reranker)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(reranker)

            logger.info(
                f"Created Reranker entity: {reranker.id} (model_id: {reranker.model_id})"
            )
            return reranker

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating Reranker: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"模型ID '{reranker_data.model_id}' 或模型名称 '{reranker_data.model_name}' 已经存在。"
                ) from e
            else:
                raise ValueError(f"模型创建失败: {e}") from e

    async def update_reranker(
        self, reranker_id: str, update_data: RerankerModelCreate
    ) -> RerankerModelEntity:
        """
        Update an existing Reranker entity.
        Note: Caller is responsible for committing the session.

        Args:
            reranker_id: Reranker entity ID
            update_data: Updated Reranker data

        Returns:
            Updated RerankerModelEntity (not yet committed)

        Raises:
            ValueError: If Reranker entity not found
        """
        reranker = await self.session.get(RerankerModelEntity, reranker_id)
        if not reranker:
            raise ValueError(f"Reranker '{reranker_id}' 不存在。")

        logger.info(f"Updating Reranker {reranker_id} with data: {update_data}")

        # Update fields
        if update_data.model_id is not None:
            reranker.model_id = update_data.model_id
        if update_data.model_name is not None:
            reranker.model_name = update_data.model_name
        if update_data.base_url is not None:
            reranker.base_url = update_data.base_url
        if update_data.type is not None:
            reranker.type = update_data.type
        if update_data.api_key is not None:
            reranker.encrypted_api_key = encrypt_key(update_data.api_key)

        self.session.add(reranker)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(reranker)

        logger.info(
            f"Updated Reranker entity: {reranker.id} (model_id: {reranker.model_id})"
        )
        return reranker

    async def delete_reranker(self, reranker_id: str) -> None:
        """
        Delete a Reranker entity.
        Note: Caller is responsible for committing the session.

        Args:
            reranker_id: Reranker entity ID

        Raises:
            ValueError: If Reranker entity not found
        """
        reranker = await self.session.get(RerankerModelEntity, reranker_id)
        if not reranker:
            raise ValueError(f"Reranker '{reranker_id}' 不存在。")

        # Delete from database (staged, not committed)
        await self.session.delete(reranker)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(
            f"Deleted Reranker entity: {reranker_id} (model_id: {reranker.model_id})"
        )

    async def get_all_rerankers(self) -> List[RerankerModelEntity]:
        """
        Get all Reranker entities without pagination.

        Returns:
            List of all RerankerModelEntity
        """
        statement = select(RerankerModelEntity)
        results = await self.session.exec(statement)
        return list(results.all())
