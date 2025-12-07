"""Trace Service layer for database operations."""

from typing import Optional
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.trace import TraceModel, TraceModelEntity


class TraceService:
    """Service layer for Trace entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize TraceService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_trace_config(
        self,
    ) -> Optional[TraceModelEntity]:
        """
        Get the trace config entity (usually only one with id='default_trace_id').

        Returns:
            TraceModelEntity if found, None otherwise
        """
        statement = select(TraceModelEntity)
        result = await self.session.exec(statement)
        return result.first()

    async def create_or_update_trace_config(
        self, config_data: TraceModel
    ) -> TraceModelEntity:
        """
        Create or update a Trace config entity.
        Note: Caller is responsible for committing the session.

        Args:
            config_data: Trace config data

        Returns:
            Created or updated TraceModelEntity (not yet committed)
        """
        # Get existing config or create new one
        statement = select(TraceModelEntity)
        result = await self.session.exec(statement)
        config = result.first()

        if config is None:
            # Create new config
            config = TraceModelEntity.model_validate(config_data)
            self.session.add(config)
            logger.info("Creating new Trace config")
        else:
            # Update existing config
            if config_data.endpoint is not None:
                config.endpoint = config_data.endpoint
            if config_data.enabled is not None:
                config.enabled = config_data.enabled
            if config_data.token is not None:
                config.token = config_data.token
            if config_data.service_name is not None:
                config.service_name = config_data.service_name
            if config_data.user_args is not None:
                config.user_args = config_data.user_args

            self.session.add(config)
            logger.info(f"Updating Trace config: {config.id}")

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(config)

            logger.info(f"Created/Updated Trace config: {config.id}")
            return config

        except IntegrityError as e:
            logger.error(
                f"IntegrityError when creating/updating Trace config: {e.orig}"
            )
            raise ValueError(f"配置创建/更新失败: {e}") from e
