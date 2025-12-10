import re
from typing import Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from mcp.server.fastmcp import FastMCP
from db.models.knowledgebase.knowledgebase import KbEntity
from api.v1.mcp.kb_retriever_tool import get_retrieval_tool
from db.db_context import with_async_db_session, AsyncSession
from loguru import logger
import anyio
from common.system_constants import DEFAULT_TENANT_ID

# Global task group for MCP session managers
# This is set in the FastAPI lifespan context
mcp_task_group: Optional[anyio.abc.TaskGroup] = None


class KbMcpServerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically create and mount MCP servers for knowledge bases.

    When a request matches the pattern `/v1/retrieval/knowledgebases/{kb_id}`,
    this middleware ensures that an MCP server exists for that knowledge base.
    If it doesn't exist, it creates and mounts the server dynamically.
    """

    # Pattern to match /v1/retrieval/knowledgebases/{kb_id}
    PATH_PATTERN = re.compile(r'^/v1/retrieval/knowledgebases/([^/]+)')

    def __init__(self, app, fastapi_app: Optional[FastAPI] = None):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application (next middleware in the chain)
            fastapi_app: The FastAPI application instance for mounting sub-applications
        """
        super().__init__(app)
        self._fastapi_app = fastapi_app
        self._servers: Dict[str, FastMCP] = {}
        self._lock = anyio.Lock()


    @with_async_db_session
    async def ensure_mcp_exists_async(self, kb_id: str, tenant_id: str, session: AsyncSession) -> None:
        """
        Ensure MCP server exists for the given knowledge base ID.

        This method is thread-safe and will only create the server once,
        even if called concurrently.

        Args:
            kb_id: The knowledge base ID
            session: Database session
            tenant_id: The tenant ID
        Raises:
            ValueError: If the knowledge base is not found
            RuntimeError: If FastAPI app or task group is not available
        """
        # Fast path: check without lock first
        if kb_id in self._servers:
            logger.debug(f"Knowledgebase {kb_id} MCP server already exists.")
            return

        # Acquire lock for thread-safe creation
        async with self._lock:
            # Double-check after acquiring lock
            if kb_id in self._servers:
                logger.debug(f"Knowledgebase {kb_id} MCP server already exists.")
                return

            try:
                logger.info(f"Creating MCP server for knowledgebase {kb_id}.")

                # Validate knowledge base exists
                kb = await session.get(KbEntity, kb_id)
                if kb is None:
                    raise ValueError(f"Knowledge base {kb_id} not found.")

                # Validate prerequisites
                if self._fastapi_app is None:
                    raise RuntimeError("FastAPI app instance not available for mounting MCP server")

                if mcp_task_group is None:
                    raise RuntimeError("MCP task group not available. Ensure FastAPI lifespan is properly configured.")

                # Create retrieval tool
                retrieval_tool = get_retrieval_tool(
                    knowledgebase_id=kb_id,
                    kb_name=kb.name,
                    kb_description=kb.description or "",
                    tenant_id=tenant_id,
                )

                # Create FastMCP server
                new_mcp = FastMCP(
                    name=f"mcp-knowledgebase-{kb.id}",
                    tools=[retrieval_tool],
                    json_response=True,
                    stateless_http=True,
                )

                # Mount the MCP server
                mount_path = f"/v1/retrieval/knowledgebases/{kb.id}"
                self._fastapi_app.mount(mount_path, new_mcp.streamable_http_app())

                # Configure session manager with global task group, must be called after streamable_http_app() since session_managed is created lazily.
                new_mcp._session_manager._task_group = mcp_task_group
                new_mcp._session_manager._has_started = True

                self._servers[kb.id] = new_mcp

                logger.info(f"Knowledgebase {kb.id} MCP server mounted at {mount_path}.")

            except Exception as e:
                logger.error(f"Failed to create MCP server for {kb_id}: {e}")
                raise



    async def dispatch(self, request: Request, call_next):
        """
        Intercept requests and ensure MCP server exists before processing.

        If the request path matches the pattern for a knowledge base MCP server,
        this method ensures the server is created and mounted before the request
        is processed.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain

        Returns:
            Response from the next handler or an error response
        """
        path = request.url.path
        tenant_id = request.headers.get("X-TENANT-ID") or DEFAULT_TENANT_ID

        # Check if the path matches the knowledge base MCP server pattern
        match = self.PATH_PATTERN.match(path)
        if not match:
            # Path doesn't match, continue with normal request processing
            return await call_next(request)

        # Extract knowledge base ID from path
        kb_id = match.group(1)

        try:
            # Ensure MCP server exists for this knowledge base
            await self.ensure_mcp_exists_async(kb_id, tenant_id=tenant_id)
        except ValueError as e:
            # Knowledge base not found
            logger.error(f"Failed to create MCP server for {kb_id}: {e}")
            return Response(
                content=f'{{"error": "{str(e)}"}}',
                status_code=404,
                media_type="application/json"
            )
        except RuntimeError as e:
            # Configuration error (missing app or task group)
            logger.error(f"Configuration error for MCP server {kb_id}: {e}")
            return Response(
                content=f'{{"error": "Server configuration error: {str(e)}"}}',
                status_code=500,
                media_type="application/json"
            )
        except Exception as e:
            # Other unexpected errors
            logger.exception(f"Unexpected error creating MCP server for {kb_id}: {e}")
            return Response(
                content=f'{{"error": "Failed to create MCP server: {str(e)}"}}',
                status_code=500,
                media_type="application/json"
            )

        # Continue with the request (MCP server will handle it)
        return await call_next(request)
