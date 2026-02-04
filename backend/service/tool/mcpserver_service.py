"""MCP Server Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger
from db.models.mcp import McpServerCreate, McpServerEntity
from common.encrypt_utils import encrypt_key
from common.chat.response_model import PagedResult


class McpserverService:
    """Service layer for MCP Server entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize McpserverService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    async def get_mcpserver(self, mcp_id: str, tenant_id: str) -> Optional[McpServerEntity]:
        """
        Get a single MCP Server entity by ID.

        Args:
            mcp_id: MCP Server entity ID

        Returns:
            McpServerEntity if found, None otherwise
        """
        result = await self.session.exec(select(McpServerEntity).where(McpServerEntity.id == mcp_id, McpServerEntity.tenant_id == tenant_id))
        return result.first()

    async def get_mcpserver_by_name(self, name: str, tenant_id: str) -> Optional[McpServerEntity]:
        """
        Get a single MCP Server entity by name.

        Args:
            name: MCP Server name

        Returns:
            McpServerEntity if found, None otherwise
        """
        statement = select(McpServerEntity).where(McpServerEntity.name == name, McpServerEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        return result.first()

    async def get_mcpserver_by_ids(self, ids: List[str], tenant_id: str) -> List[McpServerEntity]:
        """
        Get MCP Server entities by IDs.

        Args:
            ids: MCP Server entity IDs

        Returns:
            List of McpServerEntity if found, None otherwise
        """
        statement = select(McpServerEntity).where(McpServerEntity.id.in_(ids), McpServerEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        return list(result.all())

    async def list_mcpservers(
        self,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
        name: Optional[str] = None,
    ) -> PagedResult[List[McpServerEntity]]:
        """
        List MCP Server entities with pagination and optional filtering.

        Args:
            page: Page number (1-indexed)
            size: Page size
            name: Optional filter for name

        Returns:
            PagedResult containing list of McpServerEntity and pagination metadata
        """
        # Build base query
        base_query = select(McpServerEntity).where(McpServerEntity.tenant_id == tenant_id)

        # Add name filter if provided
        if name is not None:
            base_query = base_query.where(McpServerEntity.name == name, McpServerEntity.tenant_id == tenant_id)

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = base_query.offset(offset).limit(size)
        results = await self.session.exec(paginated_query)
        mcps = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=mcps,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_mcpserver(self, mcp_data: McpServerCreate, tenant_id: str) -> McpServerEntity:
        """
        Create a new MCP Server entity.
        Note: Caller is responsible for committing the session.

        Args:
            mcp_data: MCP Server creation data

        Returns:
            Created McpServerEntity (not yet committed)

        Raises:
            ValueError: If name already exists (IntegrityError converted)
        """
        # Encrypt auth token
        encrypted_auth_token = (
            encrypt_key(mcp_data.auth_token) if mcp_data.auth_token else None
        )

        # Create entity
        mcp = McpServerEntity.model_validate(
            mcp_data, update={"encrypted_auth_token": encrypted_auth_token, "tenant_id": tenant_id}
        )

        self.session.add(mcp)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(mcp)

            logger.info(f"Created MCP Server entity: {mcp.id} (name: {mcp.name})")
            return mcp

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating MCP Server: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(f"MCP name '{mcp_data.name}' already exists.") from e
            else:
                raise ValueError(f"MCP creation failed: {e}") from e

    async def update_mcpserver(
        self, mcp_id: str, update_data: McpServerCreate, tenant_id: str
    ) -> McpServerEntity:
        """
        Update an existing MCP Server entity.
        Note: Caller is responsible for committing the session.

        Args:
            mcp_id: MCP Server entity ID
            update_data: Updated MCP Server data

        Returns:
            Updated McpServerEntity (not yet committed)

        Raises:
            ValueError: If MCP Server entity not found
        """
        result = await self.session.exec(select(McpServerEntity).where(McpServerEntity.id == mcp_id, McpServerEntity.tenant_id == tenant_id))
        mcp = result.first()
        if not mcp:
            raise ValueError(f"MCP '{mcp_id}' does not exist.")

        logger.info(f"Updating MCP Server {mcp_id} with data: {update_data}")

        # Update fields
        if update_data.name is not None:
            mcp.name = update_data.name
        if update_data.enabled is not None:
            mcp.enabled = update_data.enabled
        if update_data.auth_token is not None:
            mcp.encrypted_auth_token = encrypt_key(update_data.auth_token)
        if update_data.type is not None:
            mcp.type = update_data.type
        if update_data.url is not None:
            mcp.url = update_data.url
        if update_data.need_token is not None:
            mcp.need_token = update_data.need_token

        self.session.add(mcp)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(mcp)

        logger.info(f"Updated MCP Server entity: {mcp.id} (name: {mcp.name})")
        return mcp

    async def delete_mcpserver(self, mcp_id: str, tenant_id: str) -> None:
        """
        Delete a MCP Server entity.
        Note: Caller is responsible for committing the session.

        Args:
            mcp_id: MCP Server entity ID

        Raises:
            ValueError: If MCP Server entity not found
        """
        result = await self.session.exec(select(McpServerEntity).where(McpServerEntity.id == mcp_id, McpServerEntity.tenant_id == tenant_id))
        mcp = result.first()
        if not mcp:
            raise ValueError(f"MCP '{mcp_id}' does not exist.")

        # Delete from database (staged, not committed)
        await self.session.delete(mcp)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted MCP Server entity: {mcp_id} (name: {mcp.name})")

    async def get_all_mcpservers(self, tenant_id: str) -> List[McpServerEntity]:
        """
        Get all MCP Server entities without pagination.

        Returns:
            List of all McpServerEntity
        """
        statement = select(McpServerEntity).where(McpServerEntity.tenant_id == tenant_id)
        results = await self.session.exec(statement)
        return list(results.all())
