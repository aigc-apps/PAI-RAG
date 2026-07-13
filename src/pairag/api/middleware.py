from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from asgi_correlation_id import CorrelationIdMiddleware
import time
from loguru import logger


class CustomMiddleWare:
    """Pure ASGI middleware for request timing / logging.

    The previous implementation extended Starlette's ``BaseHTTPMiddleware``,
    which buffers the response body through an anyio memory stream. That
    interfered with Server-Sent Events on ``/v1/chat/completions``: first
    tokens were held back until the full stream finished, and
    ``CancelledError`` raised on client disconnect was swallowed, leaving
    upstream LLM connections open. Both symptoms lead to intermittent 499s
    behind nginx / ALB.

    Implementing the raw ASGI interface lets us wrap ``send`` and only touch
    the ``http.response.start`` headers, so streaming chunks pass through
    untouched and cancellations propagate to the SSE generator naturally.
    """

    def __init__(self, app):
        self.app = app
        self.last_log_time = 0.0
        self.log_interval = 60  # seconds

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        client = scope.get("client")
        host = client[0] if client else "unknown"

        method = scope.get("method", "")
        path = scope.get("path", "")
        query_string = scope.get("query_string", b"")
        if query_string:
            try:
                path = f"{path}?{query_string.decode('latin-1')}"
            except Exception:
                pass

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                process_time = time.time() - start_time
                headers = list(message.get("headers", []))
                headers.append(
                    (b"x-process-time", f"{process_time:.4f}".encode("latin-1"))
                )
                headers.append((b"x-client-ip", host.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            process_time = time.time() - start_time
            log_line = (
                f"Request: {method} {path} - Response Time: {process_time:.4f} "
                f"seconds Host {host}"
            )
            if "get_upload_state" in path:
                current_time = time.time()
                if current_time - self.last_log_time >= self.log_interval:
                    logger.info(log_line)
                    self.last_log_time = current_time
            else:
                logger.info(log_line)


def _configure_session_middleware(app):
    app.add_middleware(
        CorrelationIdMiddleware,
        header_name="X-Request-ID",
    )


def _configure_cors_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )


def add_middlewares(app: FastAPI):
    # reset current middleware to allow modifying user provided list
    app.middleware_stack = None
    _configure_cors_middleware(app)
    _configure_session_middleware(app)
    app.add_middleware(CustomMiddleWare)
    app.build_middleware_stack()  # rebuild middleware stack on-the-fly
