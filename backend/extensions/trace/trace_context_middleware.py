# backend/middleware/trace_context_advanced.py

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry.propagate import get_global_textmap
from opentelemetry import context
from opentelemetry.baggage import get_baggage
import uuid
import os
from loguru import logger

from extensions.trace.context import (
    AGENTSCOPE_REQUEST_ID_KEY,
    set_request_id,
)

ENABLE_TRACE_CONTEXT_DEBUG = os.getenv("ENABLE_TRACE_CONTEXT_DEBUG", "false").lower() in ["true", "1", "yes", "y"]


class TraceContextMiddleware(BaseHTTPMiddleware):
    """
    提取 Trace Context 并设置到日志上下文
    """

    async def dispatch(self, request: Request, call_next):
        # 提取 trace context
        carrier = dict(request.headers)
        if ENABLE_TRACE_CONTEXT_DEBUG:
            logger.info(f"Trace context debug headers: {carrier}")
        propagator = get_global_textmap()
        extracted_context = propagator.extract(carrier=carrier)

        # 在提取的 context 中执行
        token = context.attach(extracted_context)
        try:
            # 从 baggage 中获取 request_id 并设置到 context
            request_id = get_baggage(AGENTSCOPE_REQUEST_ID_KEY) or uuid.uuid4().hex
            set_request_id(request_id)

            response = await call_next(request)
            return response
        finally:
            # 清理 request_id
            set_request_id(None)
            context.detach(token)
