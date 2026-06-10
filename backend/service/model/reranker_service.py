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

    async def get_reranker(self, reranker_id: str, tenant_id: str) -> Optional[RerankerModelEntity]:
        """
        Get a single Reranker entity by ID.

        Args:
            reranker_id: Reranker entity ID

        Returns:
            RerankerModelEntity if found, None otherwise
        """
        result = await self.session.exec(select(RerankerModelEntity).where(RerankerModelEntity.id == reranker_id, RerankerModelEntity.tenant_id == tenant_id))
        return result.first()

    async def get_reranker_by_model_id(
        self, model_id: str, tenant_id: str
    ) -> Optional[RerankerModelEntity]:
        """
        Get a single Reranker entity by model_id.

        Args:
            model_id: Reranker model_id

        Returns:
            RerankerModelEntity if found, None otherwise
        """
        statement = select(RerankerModelEntity).where(
            RerankerModelEntity.model_id == model_id, RerankerModelEntity.tenant_id == tenant_id
        )
        result = await self.session.exec(statement)
        return result.first()

    async def get_reranker_by_model_name(
        self, model_name: str, tenant_id: str
    ) -> Optional[RerankerModelEntity]:
        """
        Get a single Reranker entity by model_name.

        Args:
            model_name: Reranker model_name

        Returns:
            RerankerModelEntity if found, None otherwise
        """
        statement = select(RerankerModelEntity).where(
            RerankerModelEntity.model_name == model_name, RerankerModelEntity.tenant_id == tenant_id
        )
        result = await self.session.exec(statement)
        return result.first()

    async def list_rerankers(
        self,
        tenant_id: str,
        provider_name: Optional[str] = None,
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
        base_query = select(RerankerModelEntity).where(RerankerModelEntity.tenant_id == tenant_id)

        if provider_name is not None:
            base_query = base_query.where(
                RerankerModelEntity.provider_name == provider_name
            )

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

    async def get_provider_names(self, tenant_id: str) -> List[str]:
        """
        Get distinct provider names for rerankers.

        Args:
            tenant_id: Tenant ID

        Returns:
            List of distinct provider names
        """
        statement = select(RerankerModelEntity.provider_name).where(
            RerankerModelEntity.tenant_id == tenant_id
        ).distinct()
        result = await self.session.exec(statement)
        providers = [p for p in result.all() if p]
        # Add default if not present
        if not providers or "openai_like" not in providers:
            providers.append("openai_like")
        return sorted(set(providers))

    async def create_reranker(
        self, reranker_data: RerankerModelCreate, tenant_id: str
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
            reranker_data, update={"encrypted_api_key": encrypted_api_key, "tenant_id": tenant_id}
        )

        if reranker.provider_name is None:
            reranker.provider_name = reranker.type

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
                    f"Model ID '{reranker_data.model_id}' or model name '{reranker_data.model_name}' already exists."
                ) from e
            else:
                raise ValueError(f"Fail to create reranker: {e}") from e

    async def update_reranker(
        self, reranker_id: str, update_data: RerankerModelCreate, tenant_id: str
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
        result = await self.session.exec(select(RerankerModelEntity).where(RerankerModelEntity.id == reranker_id, RerankerModelEntity.tenant_id == tenant_id))
        reranker = result.first()
        if not reranker:
            raise ValueError(f"Reranker '{reranker_id}' does not exist.")

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
        if update_data.is_multimodal is not None:
            reranker.is_multimodal = update_data.is_multimodal
        if update_data.api_key is not None:
            reranker.encrypted_api_key = encrypt_key(update_data.api_key)
        if update_data.provider_name is not None:
            reranker.provider_name = update_data.provider_name

        if reranker.provider_name is None:
            reranker.provider_name = reranker.type
        self.session.add(reranker)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(reranker)

        logger.info(
            f"Updated Reranker entity: {reranker.id} (model_id: {reranker.model_id})"
        )
        return reranker

    async def delete_reranker(self, reranker_id: str, tenant_id: str) -> None:
        """
        Delete a Reranker entity.
        Note: Caller is responsible for committing the session.

        Args:
            reranker_id: Reranker entity ID

        Raises:
            ValueError: If Reranker entity not found
        """
        result = await self.session.exec(select(RerankerModelEntity).where(RerankerModelEntity.id == reranker_id, RerankerModelEntity.tenant_id == tenant_id))
        reranker = result.first()
        if not reranker:
            raise ValueError(f"Reranker '{reranker_id}' does not exist.")

        # Delete from database (staged, not committed)
        await self.session.delete(reranker)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(
            f"Deleted Reranker entity: {reranker_id} (model_id: {reranker.model_id})"
        )

    async def get_all_rerankers(self, tenant_id: str) -> List[RerankerModelEntity]:
        """
        Get all Reranker entities without pagination.

        Returns:
            List of all RerankerModelEntity
        """
        statement = select(RerankerModelEntity).where(RerankerModelEntity.tenant_id == tenant_id)
        results = await self.session.exec(statement)
        for reranker in results:
            if reranker.provider_name is None:
                reranker.provider_name = reranker.type
        return list(results.all())

    async def get_reranker_model_by_provider_model_id(self, provider_name: str, model_id: str, tenant_id: str) -> Optional[RerankerModelEntity]:
        """
        Get a Reranker entity by provider and model id.

        Args:
            provider_name: Reranker provider name
            model_id: Reranker model_id
            tenant_id: Tenant id

        Returns:
            RerankerModelEntity if found, None otherwise
        """
        logger.info(f"Getting Reranker model {model_id} by provider {provider_name} and tenant {tenant_id}.")
        statement = select(RerankerModelEntity).where(
            RerankerModelEntity.model_id == model_id,
            RerankerModelEntity.tenant_id == tenant_id
        )
        # if provider_name:
        #     statement = statement.where(RerankerModelEntity.provider_name == provider_name)
        # 为了兼容旧数据，不指定 provider_name 查询。直接通过model_id查询， 因为model_id有唯一性
        reranker_entity = await self.session.exec(statement)
        reranker = reranker_entity.first()
        return reranker
