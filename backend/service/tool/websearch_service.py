"""WebSearch Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.websearch import (
    WebSearchConfigCreate,
    WebSearchConfigEntity,
)
from common.encrypt_utils import encrypt_key


class WebsearchService:
    """Service layer for WebSearch entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize WebsearchService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_websearch_config(
        self, config_id: str, tenant_id: str
    ) -> Optional[WebSearchConfigEntity]:
        """
        Get a single WebSearch config entity by ID.

        Args:
            config_id: WebSearch config entity ID

        Returns:
            WebSearchConfigEntity if found, None otherwise
        """
        result = await self.session.exec(select(WebSearchConfigEntity).where(WebSearchConfigEntity.id == config_id, WebSearchConfigEntity.tenant_id == tenant_id))
        return result.first()

    async def get_websearch_config_or_create(
        self,
        tenant_id: str,
    ) -> WebSearchConfigEntity:
        """
        Get the first WebSearch config entity, or create a new one if none exists.
        Note: Caller is responsible for committing the session.

        Returns:
            WebSearchConfigEntity (existing or newly created, not yet committed)
        """
        statement = select(WebSearchConfigEntity).where(WebSearchConfigEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        config = result.first()

        return config

    async def get_all_websearch_configs(self, tenant_id: str) -> List[WebSearchConfigEntity]:
        """
        Get all WebSearch config entities (usually only one).

        Returns:
            List of all WebSearchConfigEntity
        """
        statement = select(WebSearchConfigEntity).where(WebSearchConfigEntity.tenant_id == tenant_id)
        results = await self.session.exec(statement)
        return list(results.all())

    async def create_or_update_websearch_config(
        self, config_data: WebSearchConfigCreate, tenant_id: str
    ) -> WebSearchConfigEntity:
        """
        Create or update a WebSearch config entity.
        Note: Caller is responsible for committing the session.

        Args:
            config_data: WebSearch config data

        Returns:
            Created or updated WebSearchConfigEntity (not yet committed)

        Raises:
            ValueError: If type is not supported
        """
        if config_data.type not in ["tavily", "aliyun"]:
            raise ValueError("不支持的搜索引擎类型，仅支持tavily和aliyun")

        # Encrypt keys
        encrypted_access_key_id = (
            encrypt_key(config_data.access_key_id)
            if config_data.access_key_id
            else None
        )
        encrypted_access_key_secret = (
            encrypt_key(config_data.access_key_secret)
            if config_data.access_key_secret
            else None
        )
        encrypted_tavily_api_key = (
            encrypt_key(config_data.tavily_api_key)
            if config_data.tavily_api_key
            else None
        )

        # Get existing config or create new one
        statement = select(WebSearchConfigEntity).where(WebSearchConfigEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        config = result.first()

        if config is None:
            # Create new config
            config = WebSearchConfigEntity.model_validate(
                config_data,
                update={
                    "encrypted_access_key_id": encrypted_access_key_id,
                    "encrypted_access_key_secret": encrypted_access_key_secret,
                    "encrypted_tavily_api_key": encrypted_tavily_api_key,
                    "tenant_id": tenant_id,
                },
            )
            self.session.add(config)
            logger.info("Creating new WebSearch config")
        else:
            # Update existing config
            if encrypted_access_key_id is not None:
                config.encrypted_access_key_id = encrypted_access_key_id
            if encrypted_access_key_secret is not None:
                config.encrypted_access_key_secret = encrypted_access_key_secret
            if encrypted_tavily_api_key is not None:
                config.encrypted_tavily_api_key = encrypted_tavily_api_key
            if config_data.search_count is not None:
                config.search_count = config_data.search_count
            if config_data.type is not None:
                config.type = config_data.type
            if config_data.endpoint is not None:
                config.endpoint = config_data.endpoint

            self.session.add(config)
            logger.info(f"Updating WebSearch config: {config.id}")

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(config)

            logger.info(f"Created/Updated WebSearch config: {config.id}")
            return config

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating/updating WebSearch config: {e.orig}")
            raise ValueError(f"Websearch config creation/update failed: {e}") from e

    async def delete_websearch_config(self, config_id: str, tenant_id: str) -> None:
        """
        Delete a WebSearch config entity.
        Note: Caller is responsible for committing the session.

        Args:
            config_id: WebSearch config entity ID

        Raises:
            ValueError: If WebSearch config entity not found
        """
        result = await self.session.exec(select(WebSearchConfigEntity).where(WebSearchConfigEntity.id == config_id, WebSearchConfigEntity.tenant_id == tenant_id))
        config = result.first()
        if not config:
            raise ValueError(f"WebSearch config '{config_id}' does not exist.")

        # Delete from database (staged, not committed)
        await self.session.delete(config)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted WebSearch config: {config_id}")
