from fastapi import Request, Response
from pai_rag.app.web.filebrower.constants import (
    DEFAULT_FILE_BROWER_PORT,
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


async def postprocess_middleware_to_filebrower(session, request, url):
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
    if "/filebrowser" in request.url.path and not request.url.path.endswith(
        ".DS_Store"
    ):
        url = request.url.replace(hostname="localhost", port=DEFAULT_FILE_BROWER_PORT)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=500 * 60,
                connect=500 * 60,
            )
        ) as session:
            return await postprocess_middleware_to_filebrower(session, request, url)
    else:
        response = await call_next(request)
        return response
