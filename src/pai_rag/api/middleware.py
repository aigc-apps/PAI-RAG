from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from asgi_correlation_id import CorrelationIdMiddleware
import time
from loguru import logger


class CustomMiddleWare(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.last_log_time = 0
        self.log_interval = 60  # seconds

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        host = request.client.host
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Client-IP"] = host

        if "get_upload_state" in str(request.url):
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                logger.info(
                    f"Request: {request.method} {request.url} - Response Time: {process_time:.4f} seconds Host {host}"
                )
                self.last_log_time = current_time
        else:
            logger.info(
                f"Request: {request.method} {request.url} - Response Time: {process_time:.4f} seconds Host {host}"
            )
        return response


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
