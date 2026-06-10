"""LLM Service layer for database operations."""
from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from db.models.llm import LlmModelCreate, LlmModelEntity
from common.encrypt_utils import encrypt_key
from common.chat.response_model import PagedResult
from common.llm.models import model_provider_map, llm_url_to_model_provider_id_map
from loguru import logger



class LlmService:
    """Service layer for LLM entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize LlmService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_llm(self, llm_id: str, tenant_id: str) -> Optional[LlmModelEntity]:
        """
        Get a single LLM entity by ID.

        Args:
            llm_id: LLM entity ID

        Returns:
            LlmModelEntity if found, None otherwise
        """
        result = await self.session.exec(select(LlmModelEntity).where(LlmModelEntity.id == llm_id, LlmModelEntity.tenant_id == tenant_id))
        return result.first()

    async def get_multimodal_llm(self, tenant_id: str) -> Optional[LlmModelEntity]:
        """
        Get the multimodal LLM entity.
        """
        statement = (
            select(LlmModelEntity)
            .where(
                LlmModelEntity.vision_support,
                LlmModelEntity.enabled,
                LlmModelEntity.tenant_id == tenant_id,
            )
            .order_by(
                LlmModelEntity.enabled.desc(),
                LlmModelEntity.provider_name.asc(),
                LlmModelEntity.model_id.asc(),
                LlmModelEntity.id.asc(),
            )
        )
        result = (await self.session.exec(statement)).first()
        return result


    async def get_llm_by_model_id(self, model_id: str, tenant_id: str) -> Optional[LlmModelEntity]:
        """
        Get a single LLM entity by model_id.

        Args:
            model_id: LLM model_id

        Returns:
            LlmModelEntity if found, None otherwise
        """
        statement = select(LlmModelEntity).where(
            LlmModelEntity.model_id == model_id, LlmModelEntity.tenant_id == tenant_id
        )
        result = await self.session.exec(statement)
        return result.first()

    async def list_llms(
        self,
        tenant_id: str,
        provider_name: Optional[str] = None,
        page: int = 1,
        size: int = 10,
        vision_support: Optional[bool] = None,
    ) -> PagedResult[List[LlmModelEntity]]:
        """
        List LLM entities with pagination and optional filtering.

        Args:
            page: Page number (1-indexed)
            size: Page size
            vision_support: Optional filter for vision support

        Returns:
            PagedResult containing list of LlmModelEntity and pagination metadata
        """
        # Build base query
        base_query = select(LlmModelEntity).where(LlmModelEntity.tenant_id == tenant_id)

        if provider_name is not None:
            base_query = base_query.where(
                LlmModelEntity.provider_name == provider_name
            )

        # Add vision_support filter if provided
        if vision_support is not None:
            base_query = base_query.where(
                LlmModelEntity.vision_support == vision_support
            )

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = base_query.offset(offset).limit(size)
        results = await self.session.exec(paginated_query)
        llms = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=llms,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def get_provider_names(self, tenant_id: str, vision_support: Optional[bool] = None) -> List[str]:
        """
        Get distinct provider names for LLMs.

        Args:
            tenant_id: Tenant ID
            vision_support: Optional filter for vision support

        Returns:
            List of distinct provider names
        """
        base_query = select(LlmModelEntity.provider_name).where(
            LlmModelEntity.tenant_id == tenant_id
        )
        if vision_support is not None:
            base_query = base_query.where(LlmModelEntity.vision_support == vision_support)
        statement = base_query.distinct()
        result = await self.session.exec(statement)
        providers = [p for p in result.all() if p]
        # Add default if not present
        if not providers or "openai_like" not in providers:
            providers.append("openai_like")
        return sorted(set(providers))

    async def create_llm(self, llm_data: LlmModelCreate, tenant_id: str) -> LlmModelEntity:
        """
        Create a new LLM entity.
        Note: Caller is responsible for committing the session.

        Args:
            llm_data: LLM creation data

        Returns:
            Created LlmModelEntity (not yet committed)

        Raises:
            ValueError: If model_id already exists (IntegrityError converted)
        """
        # Encrypt API key
        encrypted_api_key = encrypt_key(llm_data.api_key) if llm_data.api_key else None
        if llm_data.model_name is None:
            llm_data.model_name = llm_data.model # model will be deprecated, keep consistency with embedding rerank
        if llm_data.provider_name is None:
            llm_data.provider_name = llm_url_to_model_provider_id_map.get(llm_data.base_url, "openai_like")
        if llm_data.provider_name not in model_provider_map:
            raise ValueError(f"LLM creation failed: 'provider_name {llm_data.provider_name} not supported'.")

        llm_data.source = llm_data.provider_name

        # Create entity
        llm = LlmModelEntity.model_validate(
            llm_data, update={"encrypted_api_key": encrypted_api_key, "tenant_id": tenant_id}
        )

        self.session.add(llm)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(llm)

            logger.info(f"Created LLM entity: {llm.id} (model_id: {llm.model_id})")
            return llm

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating LLM: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"Model ID '{llm_data.model_id}' already exists."
                ) from e
            else:
                raise ValueError(f"LLM creation failed: {e}") from e

    async def update_llm(
        self, llm_id: str, update_data: LlmModelCreate, tenant_id: str
    ) -> LlmModelEntity:
        """
        Update an existing LLM entity.
        Note: Caller is responsible for committing the session.

        Args:
            llm_id: LLM entity ID
            update_data: Updated LLM data

        Returns:
            Updated LlmModelEntity (not yet committed)

        Raises:
            ValueError: If LLM entity not found
        """
        result = await self.session.exec(select(LlmModelEntity).where(LlmModelEntity.id == llm_id, LlmModelEntity.tenant_id == tenant_id))
        llm = result.first()
        if not llm:
            raise ValueError(f"LLM '{llm_id}' does not exist.")

        logger.info(f"Updating LLM {llm_id} with data: {update_data}")

        # Update fields
        if update_data.model_id is not None:
            llm.model_id = update_data.model_id
        if update_data.base_url is not None:
            llm.base_url = update_data.base_url
        if update_data.context_window is not None:
            llm.context_window = update_data.context_window
        if update_data.model is not None:
            llm.model = update_data.model
            llm.model_name = update_data.model # model will be deprecated, keep consistency with embedding rerank
        if update_data.model_name is not None:
            llm.model_name = update_data.model_name
        if update_data.temperature is not None:
            llm.temperature = update_data.temperature
        if update_data.api_key is not None:
            llm.encrypted_api_key = encrypt_key(update_data.api_key)
        if update_data.enabled is not None:
            llm.enabled = update_data.enabled
        if update_data.vision_support is not None:
            llm.vision_support = update_data.vision_support
        if update_data.enable_thinking is not None:
            llm.enable_thinking = update_data.enable_thinking
        if update_data.max_tokens is not None:
            llm.max_tokens = update_data.max_tokens

        self.session.add(llm)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(llm)

        logger.info(f"Updated LLM entity: {llm.id} (model_id: {llm.model_id})")
        return llm

    async def delete_llm(self, llm_id: str, tenant_id: str) -> None:
        """
        Delete an LLM entity.
        Note: Caller is responsible for committing the session.

        Args:
            llm_id: LLM entity ID

        Raises:
            ValueError: If LLM entity not found
        """
        result = await self.session.exec(select(LlmModelEntity).where(LlmModelEntity.id == llm_id, LlmModelEntity.tenant_id == tenant_id))
        llm = result.first()
        if not llm:
            raise ValueError(f"LLM '{llm_id}' does not exist.")

        # Delete from database (staged, not committed)
        await self.session.delete(llm)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted LLM entity: {llm_id} (model_id: {llm.model_id})")

    async def get_all_llms(self, tenant_id: str) -> List[LlmModelEntity]:
        """
        Get all LLM entities without pagination.

        Returns:
            List of all LlmModelEntity
        """
        statement = select(LlmModelEntity).where(LlmModelEntity.tenant_id == tenant_id)
        results = await self.session.exec(statement)
        return list(results.all())

    async def get_llm_model_by_provider_model_id(self, provider_name: str, model_id: str, tenant_id: str) -> Optional[LlmModelEntity]:
        """
        Get a LLM entity by provider and model id.

        Args:
            provider_name: LLM provider name
            model_id: LLM model_id
            tenant_id: Tenant id

        Returns:
            LlmModelEntity if found, None otherwise
        """
        logger.info(f"Getting LLM model {model_id} by provider {provider_name} and tenant {tenant_id}.")
        statement = select(LlmModelEntity).where(
            LlmModelEntity.model_id == model_id,
            LlmModelEntity.tenant_id == tenant_id
        )
        # if provider_name:
        #     statement = statement.where(LlmModelEntity.provider_name == provider_name)
        # 为了兼容旧数据，不指定 provider_name 查询。直接通过model_id查询， 因为model_id有唯一性
        result = await self.session.exec(statement)
        return result.first()
