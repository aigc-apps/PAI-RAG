import time
from fastapi import Request
from fastapi.responses import Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


# Helper to get real client IP (behind proxies)
def get_client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host or "unknown"


class CustomLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        method = request.method
        url = str(request.url)
        client_ip = get_client_ip(request)

        # Log incoming request
        logger.info(
            "Request started {method} {url} | IP: {client_ip}",
            method=method,
            url=url,
            client_ip=client_ip
        )

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            # Log exception
            duration = time.time() - start_time
            logger.error(
                "Request failed after {duration:.4f}s: {exception}",
                duration=duration,
                exception=str(exc)
            )
            raise exc
        else:
            # Log successful response
            duration = time.time() - start_time
            status_code = response.status_code
            logger.info(
                "Request completed {method} {url} | Status: {status_code} | "
                "Duration: {duration:.4f}s | IP: {client_ip}",
                method=method,
                url=url,
                status_code=status_code,
                duration=duration,
                client_ip=client_ip
            )
        finally:
            # You can add cleanup here if needed
            pass

        return response
