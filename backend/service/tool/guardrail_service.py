"""Guardrail Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.guardrail import (
    GuardrailConfigCreate,
    GuardrailConfigEntity,
)
from common.encrypt_utils import encrypt_key


class GuardrailService:
    """Service layer for Guardrail entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize GuardrailService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_guardrail_config(
        self, config_id: str, tenant_id: str
    ) -> Optional[GuardrailConfigEntity]:
        """
        Get a single Guardrail config entity by ID.

        Args:
            config_id: Guardrail config entity ID

        Returns:
            GuardrailConfigEntity if found, None otherwise
        """
        result = await self.session.exec(select(GuardrailConfigEntity).where(GuardrailConfigEntity.id == config_id, GuardrailConfigEntity.tenant_id == tenant_id))
        return result.first()

    async def get_guardrail_config_or_create(
        self,
        tenant_id: str,
    ) -> Optional[GuardrailConfigEntity]:
        """
        Get the first Guardrail config entity, or None if none exists.

        Returns:
            GuardrailConfigEntity if found, None otherwise
        """
        statement = select(GuardrailConfigEntity).where(GuardrailConfigEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        return result.first()

    async def get_all_guardrail_configs(
        self,
        tenant_id: str,
    ) -> List[GuardrailConfigEntity]:
        """
        Get all Guardrail config entities (usually only one).

        Returns:
            List of all GuardrailConfigEntity
        """
        statement = select(GuardrailConfigEntity).where(GuardrailConfigEntity.tenant_id == tenant_id)
        results = await self.session.exec(statement)
        return list(results.all())

    async def create_or_update_guardrail_config(
        self, config_data: GuardrailConfigCreate, tenant_id: str
    ) -> GuardrailConfigEntity:
        """
        Create or update a Guardrail config entity.
        Note: Caller is responsible for committing the session.

        Args:
            config_data: Guardrail config data

        Returns:
            Created or updated GuardrailConfigEntity (not yet committed)
        """
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

        # Get existing config or create new one
        statement = select(GuardrailConfigEntity).where(GuardrailConfigEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        config = result.first()

        if config is None:
            # Create new config
            config = GuardrailConfigEntity.model_validate(
                config_data,
                update={
                    "encrypted_access_key_id": encrypted_access_key_id,
                    "encrypted_access_key_secret": encrypted_access_key_secret,
                    "tenant_id": tenant_id,
                },
            )
            self.session.add(config)
            logger.info("Creating new Guardrail config")
        else:
            # Update existing config
            if encrypted_access_key_id is not None:
                config.encrypted_access_key_id = encrypted_access_key_id
            if encrypted_access_key_secret is not None:
                config.encrypted_access_key_secret = encrypted_access_key_secret
            if config_data.endpoint is not None:
                config.endpoint = config_data.endpoint
            if config_data.region_id is not None:
                config.region_id = config_data.region_id
            if config_data.region_name is not None:
                config.region_name = config_data.region_name

            self.session.add(config)
            logger.info(f"Updating Guardrail config: {config.id}")

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(config)

            logger.info(f"Created/Updated Guardrail config: {config.id}")
            return config

        except IntegrityError as e:
            logger.error(
                f"IntegrityError when creating/updating Guardrail config: {e.orig}"
            )
            raise ValueError(f"Fail to update guardrail config: {e}") from e
