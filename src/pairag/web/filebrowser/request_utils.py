from fastapi import Request, Response
from pairag.web.filebrowser.constants import (
    DEFAULT_FILE_BROWSER_PORT,
)
import aiohttp
from loguru import logger


def clean_headers(headers: dict, keys):
    for k in keys:
        headers.pop(k, None)
        headers.pop(str.lower(k), None)
    return headers


async def sender_data(req: Request):
    async for chunk in req.stream():
        yield chunk


async def postprocess_middleware_to_filebrowser(session, request, url):
    async with session.request(
        request.method,
        str(url),
        headers=clean_headers(dict(request.headers), ["Transfer-Encoding"]),
        params=str(request.path_params),
        data=sender_data(request),
        allow_redirects=False,
    ) as resp:
        content = await resp.content.read()
        return Response(
            content=content,
            headers=clean_headers(
                dict(resp.headers), ["Content-Encoding", "Content-Length"]
            ),
            status_code=resp.status,
        )


async def postprocess_middleware(request, call_next):
    logger.debug(f"request_path: {request.url.path} , method: {request.method}")
    if "/filebrowser" in request.url.path:
        url = request.url.replace(
            scheme="http", hostname="localhost", port=DEFAULT_FILE_BROWSER_PORT
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=500 * 60,
                connect=500 * 60,
            )
        ) as session:
            response = await postprocess_middleware_to_filebrowser(
                session, request, url
            )
            for k in [
                "Content-Length",
                "content-length",
                "Content-Encoding",
                "content-encoding",
            ]:
                if k in response.headers:
                    del response.headers[k]
            return response
    else:
        response = await call_next(request)
        return response
