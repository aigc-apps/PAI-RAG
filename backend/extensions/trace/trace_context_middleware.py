# backend/middleware/trace_context_advanced.py

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry.propagate import get_global_textmap
from opentelemetry import context

class TraceContextMiddleware(BaseHTTPMiddleware):
    """
    提取 Trace Context 并设置到日志上下文
    """

    async def dispatch(self, request: Request, call_next):
        # 提取 trace context
        carrier = dict(request.headers)
        propagator = get_global_textmap()
        print(f"carrier: {carrier}")
        extracted_context = propagator.extract(carrier=carrier)

        # 在提取的 context 中执行
        token = context.attach(extracted_context)
        try:
            response = await call_next(request)
            return response
        finally:
            context.detach(token)
