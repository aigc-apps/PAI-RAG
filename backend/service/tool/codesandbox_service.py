"""CodeSandbox Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.code_sandbox import (
    CodeSandboxConfigCreate,
    CodeSandboxConfigEntity,
)
from common.encrypt_utils import encrypt_key


class CodesandboxService:
    """Service layer for CodeSandbox entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize CodesandboxService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_codesandbox_config(
        self, config_id: str, tenant_id: str
    ) -> Optional[CodeSandboxConfigEntity]:
        """
        Get a single CodeSandbox config entity by ID.

        Args:
            config_id: CodeSandbox config entity ID

        Returns:
            CodeSandboxConfigEntity if found, None otherwise
        """
        codesandbox_configs = await self.session.exec(
            select(CodeSandboxConfigEntity).where(
                CodeSandboxConfigEntity.id == config_id, CodeSandboxConfigEntity.tenant_id == tenant_id))
        return codesandbox_configs.first()

    async def get_codesandbox_config_or_create(
        self,
        tenant_id: str,
    ) -> Optional[CodeSandboxConfigEntity]:
        """
        Get the first CodeSandbox config entity, or None if none exists.

        Returns:
            CodeSandboxConfigEntity if found, None otherwise
        """
        statement = select(CodeSandboxConfigEntity).where(CodeSandboxConfigEntity.tenant_id == tenant_id)
        codesandbox_configs = await self.session.exec(statement)
        return codesandbox_configs.first()

    async def get_all_codesandbox_configs(
        self,
        tenant_id: str,
    ) -> List[CodeSandboxConfigEntity]:
        """
        Get all CodeSandbox config entities (usually only one).

        Returns:
            List of all CodeSandboxConfigEntity
        """
        statement = select(CodeSandboxConfigEntity).where(CodeSandboxConfigEntity.tenant_id == tenant_id)
        codesandbox_configs = await self.session.exec(statement)
        return list(codesandbox_configs.all())

    async def create_or_update_codesandbox_config(
        self, config_data: CodeSandboxConfigCreate, tenant_id: str
    ) -> CodeSandboxConfigEntity:
        """
        Create or update a CodeSandbox config entity.
        Note: Caller is responsible for committing the session.

        Args:
            config_data: CodeSandbox config data

        Returns:
            Created or updated CodeSandboxConfigEntity (not yet committed)

        Raises:
            ValueError: If type is not supported
        """
        if config_data.type not in ["aliyun-fc"]:
            raise ValueError("不支持的code sandbox类型，仅支持aliyun-fc")

        # Get existing config or create new one
        statement = select(CodeSandboxConfigEntity).where(CodeSandboxConfigEntity.tenant_id == tenant_id)
        codesandbox_configs = await self.session.exec(statement)
        config = codesandbox_configs.first()

        provided_fields = config_data.model_dump(exclude_unset=True)

        encrypted_api_key = None
        if "api_key" in provided_fields:
            encrypted_api_key = (
                encrypt_key(config_data.api_key) if config_data.api_key else None
            )

        if config is None:
            config = CodeSandboxConfigEntity.model_validate(
                config_data,
                update={"encrypted_api_key": encrypted_api_key, "tenant_id": tenant_id}
            )
            self.session.add(config)
            logger.info(f"Creating new CodeSandbox config for type {config_data.type}")
        else:
            # Update existing config
            if config_data.aliyun_id is not None:
                config.aliyun_id = config_data.aliyun_id
            if config_data.interpreter_id is not None:
                config.interpreter_id = config_data.interpreter_id
            if config_data.interpreter_name is not None:
                config.interpreter_name = config_data.interpreter_name
            if config_data.type is not None:
                config.type = config_data.type
            if config_data.enabled is not None:
                config.enabled = config_data.enabled
            if config_data.timeout_default is not None:
                config.timeout_default = config_data.timeout_default
            if "api_key" in provided_fields:
                config.encrypted_api_key = encrypted_api_key

            self.session.add(config)
            logger.info(f"Updating CodeSandbox config: {config.id}")

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(config)

            logger.info(f"Created/Updated CodeSandbox config: {config.id}")
            return config

        except IntegrityError as e:
            logger.error(
                f"IntegrityError when creating/updating CodeSandbox config: {e.orig}"
            )
            raise ValueError(f"CodeSandbox config creation/update failed: {e}") from e
