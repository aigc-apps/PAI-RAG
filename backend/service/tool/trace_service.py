"""Trace Service layer for database operations."""

from typing import Optional
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger
import os
from db.models.trace import TraceModel, TraceModelEntity
from extensions.trace.base import init_instrument, TraceConfig
from common.system_constants import DEFAULT_TENANT_ID


DEFAULT_SERVICE_NAME = "pai-rag"

def _load_trace_config_from_env() -> TraceModelEntity:
    logger.info("Loading trace config from environment variables.")
    trace_endpoint = os.getenv("TRACE_ENDPOINT")
    if trace_endpoint:
        logger.info(f"Trace endpoint: {trace_endpoint}")
        return TraceModelEntity.model_validate({
            "endpoint": trace_endpoint,
            "token": os.getenv("TRACE_TOKEN",""),
            "service_name": os.getenv("TRACE_SERVICE_NAME", DEFAULT_SERVICE_NAME),
            "tenant_id": DEFAULT_TENANT_ID,
            "id": "default_trace_id",
            "enabled": True
        })
    else:
        return None


class TraceService:
    """Service layer for Trace entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize TraceService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def init_trace(self):
        exporter_type = os.getenv("TRACE_EXPORTER_TYPE", "grpc")
        config = _load_trace_config_from_env()
        if not config:
            statement = select(TraceModelEntity).where(TraceModelEntity.enabled)
            result = await self.session.exec(statement)
            config = result.first()

        if config and config.is_enabled():
            init_instrument(TraceConfig(
                endpoint=config.endpoint,
                token=config.token,
                service_name=config.service_name,
                user_args=config.user_args,
                enabled=config.enabled,
                exporter_type=exporter_type,
            ))
            logger.info("Initialized trace config.")
        else:
            logger.info("Trace config not enabled.")

    async def get_trace_config(
        self,
        tenant_id: str,
    ) -> Optional[TraceModelEntity]:
        """
        Get the trace config entity (usually only one with id='default_trace_id').

        Returns:
            TraceModelEntity if found, None otherwise
        """
        statement = select(TraceModelEntity).where(TraceModelEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        return result.first()

    async def create_or_update_trace_config(
        self, new_trace_config: TraceModel, tenant_id: str
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
        statement = select(TraceModelEntity).where(TraceModelEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        config = result.first()

        if config is None:
            # Create new config
            config = TraceModelEntity.model_validate(new_trace_config, update={"tenant_id": tenant_id})
            self.session.add(config)
            logger.info("Creating new Trace config")
        else:
            # Update existing config
            if new_trace_config.endpoint is not None:
                config.endpoint = new_trace_config.endpoint
            if new_trace_config.enabled is not None:
                config.enabled = new_trace_config.enabled
            if new_trace_config.token is not None:
                config.token = new_trace_config.token
            if new_trace_config.service_name is not None:
                config.service_name = new_trace_config.service_name
            if new_trace_config.user_args is not None:
                config.user_args = new_trace_config.user_args

            self.session.add(config)
            logger.info(f"Updating Trace config: {config.id}")

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(config)

            logger.info(f"Created/Updated Trace config: {config.id}")
            init_instrument(TraceConfig(
                endpoint=config.endpoint,
                token=config.token,
                service_name=config.service_name,
                user_args=config.user_args,
                enabled=config.enabled,
                exporter_type="grpc", # 默认grpc
            ))
            return config

        except IntegrityError as e:
            logger.error(
                f"IntegrityError when creating/updating Trace config: {e.orig}"
            )
            raise ValueError(f"Trace config creation/update failed: {e}") from e
