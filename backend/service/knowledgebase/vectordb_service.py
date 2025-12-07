"""VectorDB Service layer for database operations."""

from typing import Optional
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.vectordb import VectorDbConfig
from tools.utils.vectordb import create_vector_db_connection_from_env
from common.knowledgebase.types import SUPPORTED_VECTOR_DB_TYPES
from common.knowledgebase.constants import DEFAULT_VECTOR_ID


class VectordbService:
    """Service layer for VectorDB entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize VectordbService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_vectordb_config(
        self,
    ) -> Optional[VectorDbConfig]:
        """
        Get the vector database config entity (usually only one with id=DEFAULT_VECTOR_ID).

        Returns:
            VectorDbConfig if found, None otherwise
        """
        vector_config = await self.session.get(VectorDbConfig, DEFAULT_VECTOR_ID)
        if vector_config is None:
            # Create from environment if not exists
            connection = create_vector_db_connection_from_env()
            vector_config = VectorDbConfig(
                id=DEFAULT_VECTOR_ID,
                type=connection.type.value,
                config=connection.model_dump(),
            )
        return vector_config

    async def create_or_update_vectordb_config(
        self, config_data: VectorDbConfig
    ) -> VectorDbConfig:
        """
        Create or update a VectorDB config entity.
        Note: Caller is responsible for committing the session.

        Args:
            config_data: VectorDB config data

        Returns:
            Created or updated VectorDbConfig (not yet committed)

        Raises:
            ValueError: If type is not supported
        """
        if config_data.type not in SUPPORTED_VECTOR_DB_TYPES:
            raise ValueError(
                f"不支持的向量数据库类型，当前仅支持{','.join(SUPPORTED_VECTOR_DB_TYPES)}。"
            )

        config_data.config["type"] = config_data.type
        existing_vector_config = await self.session.get(
            VectorDbConfig, DEFAULT_VECTOR_ID
        )

        if existing_vector_config is None:
            logger.info(f"Creating new vectordb config for type {config_data.type}")

            # Handle password and sk from environment if not provided
            if not config_data.config.get("password"):
                env_connection = create_vector_db_connection_from_env()
                config_data.config["encrypted_password"] = (
                    env_connection.model_dump().get("encrypted_password")
                )
            if not config_data.config.get("sk"):
                env_connection = create_vector_db_connection_from_env()
                config_data.config["encrypted_sk"] = (
                    env_connection.model_dump().get("encrypted_sk")
                )

            existing_vector_config = VectorDbConfig(
                id=DEFAULT_VECTOR_ID,
                type=config_data.type,
                config=config_data.config,
            )
        else:
            existing_vector_config.type = config_data.type

            # Preserve encrypted fields if not provided
            if not config_data.config.get("password"):
                config_data.config["encrypted_password"] = (
                    existing_vector_config.config.get("encrypted_password")
                )
            if not config_data.config.get("sk"):
                config_data.config["encrypted_sk"] = (
                    existing_vector_config.config.get("encrypted_sk")
                )

            existing_vector_config.config = config_data.config

        self.session.add(existing_vector_config)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(existing_vector_config)

            logger.info(
                f"Created/Updated VectorDB config: {existing_vector_config.id} (type: {existing_vector_config.type})"
            )
            return existing_vector_config

        except IntegrityError as e:
            logger.error(
                f"IntegrityError when creating/updating VectorDB config: {e.orig}"
            )
            raise ValueError(f"配置创建/更新失败: {e}") from e

    async def prepare_test_config(
        self, test_config: VectorDbConfig
    ) -> VectorDbConfig:
        """
        Prepare a test config by filling in encrypted fields from existing config or environment.
        This is used for connection testing without modifying the actual config.

        Args:
            test_config: Test config data (may be missing password/sk)

        Returns:
            VectorDbConfig with encrypted fields filled in
        """
        if test_config.type == "local":
            # Local type doesn't need password/sk
            return test_config

        # Fill in missing encrypted fields from existing config or environment
        if not test_config.config.get("password"):
            existing_vector_config = await self.session.get(
                VectorDbConfig, DEFAULT_VECTOR_ID
            )
            if existing_vector_config is not None:
                test_config.config["encrypted_password"] = (
                    existing_vector_config.config.get("encrypted_password")
                )
            else:
                env_connection = create_vector_db_connection_from_env()
                test_config.config["encrypted_password"] = (
                    env_connection.model_dump().get("encrypted_password")
                )

        if not test_config.config.get("sk"):
            existing_vector_config = await self.session.get(
                VectorDbConfig, DEFAULT_VECTOR_ID
            )
            if existing_vector_config is not None:
                test_config.config["encrypted_sk"] = (
                    existing_vector_config.config.get("encrypted_sk")
                )
            else:
                env_connection = create_vector_db_connection_from_env()
                test_config.config["encrypted_sk"] = (
                    env_connection.model_dump().get("encrypted_sk")
                )

        return test_config
