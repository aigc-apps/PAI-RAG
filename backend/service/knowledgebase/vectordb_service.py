"""VectorDB Service layer for database operations."""

from typing import Optional
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from loguru import logger

from db.models.vectordb import VectorDbConfig
from tools.utils.vectordb import create_vector_db_connection_from_env
from common.knowledgebase.types import SUPPORTED_VECTOR_DB_TYPES
from common.knowledgebase.constants import DEFAULT_VECTOR_ID


def _try_create_env_connection():
    """Best-effort creation of the environment-configured vector db connection.

    ``create_vector_db_connection_from_env`` asserts that every field for the
    environment's configured db type is present. When the environment is only
    partially configured (e.g. ``VECTOR_DB_TYPE=hologres`` set for deployment
    but the credentials are meant to be supplied through the UI), those asserts
    raise and would otherwise block saving/testing a config the user provided
    explicitly. Returning ``None`` here lets callers simply skip the env-based
    fallback instead of failing the whole operation.
    """
    try:
        return create_vector_db_connection_from_env()
    except Exception as e:
        logger.warning(
            f"Could not build vector db connection from environment; "
            f"skipping environment fallback: {e}"
        )
        return None


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
        tenant_id: str,
    ) -> Optional[VectorDbConfig]:
        """
        Get the vector database config entity (usually only one with id=DEFAULT_VECTOR_ID).

        Returns:
            VectorDbConfig if found, None otherwise
        """
        result = await self.session.exec(select(VectorDbConfig).where(VectorDbConfig.id == DEFAULT_VECTOR_ID, VectorDbConfig.tenant_id == tenant_id))
        vector_config = result.first()
        if vector_config is None:
            # Create from environment if not exists. If the environment is not
            # (fully) configured, fall back to a local default so the page can
            # still load and the user can configure a db through the UI.
            connection = _try_create_env_connection()
            if connection is not None:
                vector_config = VectorDbConfig(
                    id=DEFAULT_VECTOR_ID,
                    type=connection.type.value,
                    config=connection.model_dump(),
                )
            else:
                vector_config = VectorDbConfig(
                    id=DEFAULT_VECTOR_ID,
                    type="local",
                    config={"type": "local"},
                )
        return vector_config

    async def create_or_update_vectordb_config(
        self, config_data: VectorDbConfig, tenant_id: str
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
        result = await self.session.exec(select(VectorDbConfig).where(VectorDbConfig.id == DEFAULT_VECTOR_ID, VectorDbConfig.tenant_id == tenant_id))
        existing_vector_config = result.first()

        if existing_vector_config is None:
            logger.info(f"Creating new vectordb config for type {config_data.type}")

            # Handle password and sk from environment if not provided.
            # The env fallback is best-effort: a partial/mismatched environment
            # must not block saving a config the user supplied through the UI.
            if not config_data.config.get("password"):
                env_connection = _try_create_env_connection()
                if env_connection is not None:
                    config_data.config["encrypted_password"] = (
                        env_connection.model_dump().get("encrypted_password")
                    )
            if not config_data.config.get("sk"):
                env_connection = _try_create_env_connection()
                if env_connection is not None:
                    config_data.config["encrypted_sk"] = (
                        env_connection.model_dump().get("encrypted_sk")
                    )

            existing_vector_config = VectorDbConfig(
                id=DEFAULT_VECTOR_ID,
                type=config_data.type,
                config=config_data.config,
                tenant_id=tenant_id,
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
            raise ValueError(f"Vector db config update failed: {e}") from e

    async def prepare_test_config(
        self, test_config: VectorDbConfig, tenant_id: str
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
            result = await self.session.exec(select(VectorDbConfig).where(VectorDbConfig.id == DEFAULT_VECTOR_ID, VectorDbConfig.tenant_id == tenant_id))
            existing_vector_config = result.first()

            if existing_vector_config is not None:
                test_config.config["encrypted_password"] = (
                    existing_vector_config.config.get("encrypted_password")
                )
            else:
                env_connection = _try_create_env_connection()
                if env_connection is not None:
                    test_config.config["encrypted_password"] = (
                        env_connection.model_dump().get("encrypted_password")
                    )

        if not test_config.config.get("sk"):
            result = await self.session.exec(select(VectorDbConfig).where(VectorDbConfig.id == DEFAULT_VECTOR_ID, VectorDbConfig.tenant_id == tenant_id))
            existing_vector_config = result.first()

            if existing_vector_config is not None:
                test_config.config["encrypted_sk"] = (
                    existing_vector_config.config.get("encrypted_sk")
                )
            else:
                env_connection = _try_create_env_connection()
                if env_connection is not None:
                    test_config.config["encrypted_sk"] = (
                        env_connection.model_dump().get("encrypted_sk")
                    )

        return test_config
