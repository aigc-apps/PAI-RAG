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
        self, config_id: str
    ) -> Optional[CodeSandboxConfigEntity]:
        """
        Get a single CodeSandbox config entity by ID.

        Args:
            config_id: CodeSandbox config entity ID

        Returns:
            CodeSandboxConfigEntity if found, None otherwise
        """
        return await self.session.get(CodeSandboxConfigEntity, config_id)

    async def get_codesandbox_config_or_create(
        self,
    ) -> Optional[CodeSandboxConfigEntity]:
        """
        Get the first CodeSandbox config entity, or None if none exists.

        Returns:
            CodeSandboxConfigEntity if found, None otherwise
        """
        statement = select(CodeSandboxConfigEntity)
        result = await self.session.exec(statement)
        return result.first()

    async def get_all_codesandbox_configs(
        self,
    ) -> List[CodeSandboxConfigEntity]:
        """
        Get all CodeSandbox config entities (usually only one).

        Returns:
            List of all CodeSandboxConfigEntity
        """
        statement = select(CodeSandboxConfigEntity)
        results = await self.session.exec(statement)
        return list(results.all())

    async def create_or_update_codesandbox_config(
        self, config_data: CodeSandboxConfigCreate
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
        statement = select(CodeSandboxConfigEntity)
        result = await self.session.exec(statement)
        config = result.first()

        if config is None:
            # Create new config
            config = CodeSandboxConfigEntity.model_validate(config_data)
            self.session.add(config)
            logger.info(f"Creating new CodeSandbox config for type {config_data.type}")
        else:
            # Update existing config
            if config_data.aliyun_id is not None:
                config.aliyun_id = config_data.aliyun_id
            if config_data.interpreter_id is not None:
                config.interpreter_id = config_data.interpreter_id
            if config_data.type is not None:
                config.type = config_data.type
            if config_data.enabled is not None:
                config.enabled = config_data.enabled
            if config_data.timeout_default is not None:
                config.timeout_default = config_data.timeout_default

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
            raise ValueError(f"配置创建/更新失败: {e}") from e
