"""ChatDB Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.chatdb.chatdb import ChatDbConfigEntity, ChatDbCreate
from common.encrypt_utils import encrypt_key


class ChatdbService:
    """Service layer for ChatDB entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize ChatdbService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_chatdb_config(
        self, config_id: str, tenant_id: str
    ) -> Optional[ChatDbConfigEntity]:
        """
        Get a single ChatDB config entity by ID.

        Args:
            config_id: ChatDB config entity ID

        Returns:
            ChatDbConfigEntity if found, None otherwise
        """
        chatdb_configs = await self.session.exec(select(ChatDbConfigEntity).where(ChatDbConfigEntity.id == config_id, ChatDbConfigEntity.tenant_id == tenant_id))
        return chatdb_configs.first()

    async def get_chatdb_config_or_create(
        self,
        tenant_id: str,
    ) -> Optional[ChatDbConfigEntity]:
        """
        Get the first ChatDB config entity, or None if none exists.

        Returns:
            ChatDbConfigEntity if found, None otherwise
        """
        statement = select(ChatDbConfigEntity).where(ChatDbConfigEntity.tenant_id == tenant_id)
        chatdb_configs = await self.session.exec(statement)
        return chatdb_configs.first()

    async def get_all_chatdb_configs(self, tenant_id: str) -> List[ChatDbConfigEntity]:
        """
        Get all ChatDB config entities (usually only one).

        Returns:
            List of all ChatDbConfigEntity
        """
        statement = select(ChatDbConfigEntity).where(ChatDbConfigEntity.tenant_id == tenant_id)
        chatdb_configs = await self.session.exec(statement)
        return list(chatdb_configs.all())

    async def create_or_update_chatdb_config(
        self, config_data: ChatDbCreate, tenant_id: str
    ) -> ChatDbConfigEntity:
        """
        Create or update a ChatDB config entity.
        Note: Caller is responsible for committing the session.

        Args:
            config_data: ChatDB config data

        Returns:
            Created or updated ChatDbConfigEntity (not yet committed)

        Raises:
            ValueError: If dialect is not supported
        """
        config_data.dialect = config_data.dialect.lower()

        if config_data.dialect not in ["mysql", "postgresql"]:
            raise ValueError("不支持的数据库类型，仅支持mysql和postgresql")

        # Encrypt password
        encrypted_password = (
            encrypt_key(config_data.password) if config_data.password else None
        )

        # Get existing config or create new one
        statement = select(ChatDbConfigEntity).where(ChatDbConfigEntity.tenant_id == tenant_id)
        chatdb_configs = await self.session.exec(statement)
        config = chatdb_configs.first()

        if config is None:
            # Create new config
            config = ChatDbConfigEntity.model_validate(
                config_data,
                update={"encrypted_password": encrypted_password, "tenant_id": tenant_id},
            )
            self.session.add(config)
            logger.info(f"Creating new ChatDB config for dialect {config_data.dialect}")
        else:
            # Update existing config
            config.dialect = config_data.dialect
            config.db_name = config_data.db_name
            config.username = config_data.username
            if encrypted_password is not None:
                config.encrypted_password = encrypted_password
            config.model_id = config_data.model_id
            config.host = config_data.host
            config.port = config_data.port

            self.session.add(config)
            logger.info(f"Updating ChatDB config: {config.id}")

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(config)

            logger.info(f"Created/Updated ChatDB config: {config.id}")
            return config

        except IntegrityError as e:
            logger.error(
                f"IntegrityError when creating/updating ChatDB config: {e.orig}"
            )
            raise ValueError(f"Fail to update chatdb config: {e}") from e
