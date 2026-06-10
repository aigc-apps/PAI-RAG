"""Embedding Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.knowledgebase.embedding import (
    EmbeddingModelCreate,
    EmbeddingModelEntity,
    EmbeddingType,
)
from common.encrypt_utils import encrypt_key
from common.chat.response_model import PagedResult
from service.factory.model_factory import create_embedding_model
from llama_index.core.embeddings import BaseEmbedding
from common.knowledgebase.constants import DEFAULT_EMBEDDING_MODEL

class EmbeddingService:
    """Service layer for Embedding entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize EmbeddingService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_embedding(self, emb_id: str, tenant_id: str) -> Optional[EmbeddingModelEntity]:
        """
        Get a single Embedding entity by ID.

        Args:
            emb_id: Embedding entity ID

        Returns:
            EmbeddingModelEntity if found, None otherwise
        """
        result = await self.session.exec(select(EmbeddingModelEntity).where(EmbeddingModelEntity.id == emb_id, EmbeddingModelEntity.tenant_id == tenant_id))
        return result.first()

    async def get_embedding_by_model_id(
        self, model_id: str, tenant_id: str
    ) -> Optional[EmbeddingModelEntity]:
        """
        Get a single Embedding entity by model_id.

        Args:
            model_id: Embedding model_id

        Returns:
            EmbeddingModelEntity if found, None otherwise
        """
        statement = select(EmbeddingModelEntity).where(
            EmbeddingModelEntity.model_id == model_id, EmbeddingModelEntity.tenant_id == tenant_id
        )
        result = await self.session.exec(statement)
        return result.first()


    async def get_default_embedding(self, tenant_id: str) -> Optional[EmbeddingModelEntity]:
        """
        Get the default Embedding entity.

        Returns:
            EmbeddingModelEntity if found, None otherwise
        """
        statement = select(EmbeddingModelEntity).where(EmbeddingModelEntity.model_id == "BAAI/bge-m3", EmbeddingModelEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        default_embedding = result.first()
        if not default_embedding:
            logger.info(f"No default embedding model was found, and using {DEFAULT_EMBEDDING_MODEL} for attachment knowledgebase.")
            default_embedding = await self.create_embedding(
                embedding_data=EmbeddingModelCreate(
                    tenant_id=tenant_id,
                    model_name=DEFAULT_EMBEDDING_MODEL,
                    model_id=DEFAULT_EMBEDDING_MODEL,
                    dimension=1024,
                    type=EmbeddingType.LOCAL,
                    provider_name="openai_like",
                    is_default=True,
                    is_ready=True,
                ),
                tenant_id=tenant_id,
            )
            await self.session.commit()
        return default_embedding


    async def get_embedding_by_model_name(
        self, model_name: str, tenant_id: str
    ) -> Optional[EmbeddingModelEntity]:
        """
        Get a single Embedding entity by model_name.

        Args:
            model_name: Embedding model_name

        Returns:
            EmbeddingModelEntity if found, None otherwise
        """
        statement = select(EmbeddingModelEntity).where(
            EmbeddingModelEntity.model_name == model_name, EmbeddingModelEntity.tenant_id == tenant_id
        )
        result = await self.session.exec(statement)
        embedding = result.first()
        if embedding and not embedding.provider_name:
            embedding.provider_name = "openai_like"
        return embedding

    async def list_embeddings(
        self,
        tenant_id: str,
        provider_name: Optional[str] = None,
        page: int = 1,
        size: int = 10,
        model_name: Optional[str] = None,
    ) -> PagedResult[List[EmbeddingModelEntity]]:
        """
        List Embedding entities with pagination and optional filtering.

        Args:
            page: Page number (1-indexed)
            size: Page size
            model_name: Optional filter for model_name

        Returns:
            PagedResult containing list of EmbeddingModelEntity and pagination metadata
        """
        # Build base query
        base_query = select(EmbeddingModelEntity).where(EmbeddingModelEntity.tenant_id == tenant_id)

        if provider_name is not None:
            base_query = base_query.where(
                EmbeddingModelEntity.provider_name == provider_name
            )
        # Add model_name filter if provided
        if model_name is not None:
            base_query = base_query.where(
                EmbeddingModelEntity.model_name == model_name
            )

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = base_query.offset(offset).limit(size)
        results = await self.session.exec(paginated_query)
        embeddings = list(results.all())
        for embedding in embeddings:
            if not embedding.provider_name:
                embedding.provider_name = "openai_like"
        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=embeddings,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def get_provider_names(self, tenant_id: str) -> List[str]:
        """
        Get distinct provider names for embeddings.

        Args:
            tenant_id: Tenant ID

        Returns:
            List of distinct provider names
        """
        statement = select(EmbeddingModelEntity.provider_name).where(
            EmbeddingModelEntity.tenant_id == tenant_id
        ).distinct()
        result = await self.session.exec(statement)
        providers = [p for p in result.all() if p]
        # Add default if not present
        if not providers or "openai_like" not in providers:
            providers.append("openai_like")
        return sorted(set(providers))

    async def create_embedding(
        self, embedding_data: EmbeddingModelCreate, tenant_id: str
    ) -> EmbeddingModelEntity:
        """
        Create a new Embedding entity.
        Note: Caller is responsible for committing the session.

        Args:
            embedding_data: Embedding creation data

        Returns:
            Created EmbeddingModelEntity (not yet committed)

        Raises:
            ValueError: If model_id already exists (IntegrityError converted)
        """
        # Encrypt API key
        encrypted_api_key = (
            encrypt_key(embedding_data.api_key) if embedding_data.api_key else None
        )
        if embedding_data.provider_name is None:
            embedding_data.provider_name = "openai_like"
        # Create entity
        embedding = EmbeddingModelEntity.model_validate(
            embedding_data, update={"encrypted_api_key": encrypted_api_key, "tenant_id": tenant_id}
        )

        # Set is_ready based on type
        if embedding.type == EmbeddingType.OPENAI_LIKE:
            embedding.is_ready = True

        self.session.add(embedding)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(embedding)

            logger.info(
                f"Created Embedding entity: {embedding.id} (model_id: {embedding.model_id})"
            )
            return embedding

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating Embedding: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"Model ID '{embedding_data.model_id}' or model name '{embedding_data.model_name}' already exists."
                ) from e
            else:
                raise ValueError(f"Embedding creation failed: {e}") from e

    async def update_embedding(
        self, emb_id: str, update_data: EmbeddingModelCreate, tenant_id: str
    ) -> EmbeddingModelEntity:
        """
        Update an existing Embedding entity.
        Note: Caller is responsible for committing the session.

        Args:
            emb_id: Embedding entity ID
            update_data: Updated Embedding data

        Returns:
            Updated EmbeddingModelEntity (not yet committed)

        Raises:
            ValueError: If Embedding entity not found
        """
        result = await self.session.exec(select(EmbeddingModelEntity).where(EmbeddingModelEntity.id == emb_id, EmbeddingModelEntity.tenant_id == tenant_id))
        embedding = result.first()
        if not embedding:
            raise ValueError(f"Embedding '{emb_id}' does not exist.")

        logger.info(f"Updating Embedding {emb_id} with data: {update_data}")

        # Update fields
        if update_data.model_name is not None:
            embedding.model_name = update_data.model_name
        if update_data.dimension is not None:
            embedding.dimension = update_data.dimension
        if update_data.type is not None:
            embedding.type = update_data.type
        if update_data.endpoint is not None:
            embedding.endpoint = update_data.endpoint
        if update_data.is_ready is not None:
            embedding.is_ready = update_data.is_ready
        if update_data.embed_batch_size is not None:
            embedding.embed_batch_size = update_data.embed_batch_size
        if update_data.is_default is not None:
            embedding.is_default = update_data.is_default
        if update_data.is_multimodal is not None:
            embedding.is_multimodal = update_data.is_multimodal
        if update_data.api_key is not None:
            embedding.encrypted_api_key = encrypt_key(update_data.api_key)
        if update_data.provider_name is not None:
            embedding.provider_name = update_data.provider_name
        if embedding.provider_name is None:
            embedding.provider_name = "openai_like"

        self.session.add(embedding)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(embedding)

        logger.info(
            f"Updated Embedding entity: {embedding.id} (model_id: {embedding.model_id})"
        )
        return embedding

    async def delete_embedding(self, emb_id: str, tenant_id: str) -> None:
        """
        Delete an Embedding entity.
        Note: Caller is responsible for committing the session.

        Args:
            emb_id: Embedding entity ID

        Raises:
            ValueError: If Embedding entity not found
        """
        result = await self.session.exec(select(EmbeddingModelEntity).where(EmbeddingModelEntity.id == emb_id, EmbeddingModelEntity.tenant_id == tenant_id))
        embedding = result.first()

        if not embedding:
            raise ValueError(f"Embedding '{emb_id}' by tenant {tenant_id} not found.")

        assert embedding.tenant_id == tenant_id, f"Embedding '{emb_id}' by tenant {tenant_id} not found."

        # Delete from database (staged, not committed)
        await self.session.delete(embedding)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(
            f"Deleted Embedding entity: {emb_id} (model_id: {embedding.model_id})"
        )

    async def get_all_embeddings(self, tenant_id: str) -> List[EmbeddingModelEntity]:
        """
        Get all Embedding entities without pagination.

        Returns:
            List of all EmbeddingModelEntity
        """
        statement = select(EmbeddingModelEntity).where(EmbeddingModelEntity.tenant_id == tenant_id)
        results = await self.session.exec(statement)
        embeddings = list(results.all())
        for embedding in embeddings:
            if not embedding.provider_name:
                embedding.provider_name = "openai_like"
        return embeddings

    async def get_embedding_model(self, model_id: str, tenant_id: str) -> Optional[BaseEmbedding]:
        """
        Get an Embedding entity by model_id.

        Args:
            model_id: Embedding model_id

        Returns:
            EmbeddingModelEntity if found, None otherwise
        """
        embedding_entity = await self.get_embedding_by_model_id(model_id, tenant_id)
        if not embedding_entity:
            raise ValueError(f"Embedding model {model_id} not found.")

        if not embedding_entity.provider_name:
            embedding_entity.provider_name = "openai_like"
        return create_embedding_model(embedding_entity)

    async def get_embedding_model_by_provider_model_id(self, provider_name: str, model_id: str, tenant_id: str) -> Optional[EmbeddingModelEntity]:
        """
        Get an Embedding entity by provider and model id.

        Args:
            provider_name: Embedding model provider name
            model_id: Embedding model id
            tenant_id: Tenant id

        Returns:
            EmbeddingModelEntity if found, None otherwise
        """
        logger.info(f"Getting embedding model by provider {provider_name} and model id {model_id} for tenant {tenant_id}.")
        statement = select(EmbeddingModelEntity).where(
            EmbeddingModelEntity.model_id == model_id,
            EmbeddingModelEntity.tenant_id == tenant_id
        )
        # if provider_name:
        #     statement = statement.where(EmbeddingModelEntity.provider_name == provider_name)
        # 为了兼容旧数据，不指定 provider_name 查询。直接通过model_id查询， 因为model_id有唯一性
        embedding_entity = await self.session.exec(statement)
        embedding = embedding_entity.first()
        return embedding
