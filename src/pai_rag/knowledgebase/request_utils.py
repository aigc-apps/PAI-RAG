from fastapi import Request, Response
from pai_rag.knowledgebase.constants import DEFAULT_KNOWLEDGE_PATH
from pai_rag.core.service_daemon import batch_files, batch_lock
from pai_rag.app.web.tabs.model.index_info import delete_index
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


async def is_directory_or_file(session, url, headers):
    async with session.get(
        str(url),
        headers=clean_headers(dict(headers), ["Transfer-Encoding"]),
    ) as resp:
        if resp.status == 200:
            data = await resp.json()
            logger.debug(f"resp data: {data}")
            return data.get("isDir"), data.get("name"), data.get("path").split("/")[1]
        else:
            logger.error(f"请求失败，状态码: {resp.status}")
            return False, None, None


async def postprocess_middleware(request, call_next):
    logger.debug(f"request_path: {request.url.path} , method: {request.method}")
    if "/filebrowser" in request.url.path:
        url = request.url.replace(hostname="localhost", port="8012")
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=500 * 60,
                connect=500 * 60,
            )
        ) as session:
            if request.method.lower() == "delete":
                is_dir, file_name, index_name = await is_directory_or_file(
                    session, url, request.headers
                )
                if is_dir:
                    delete_index(index_name)
                    logger.info(f"Index Deleted. Index name: {index_name}")
                else:
                    logger.info(
                        f"File Deleted. File name: {file_name} from index {index_name}"
                    )
            elif (
                "/filebrowser/api/resources" in request.url.path
                and request.method.lower() == "post"
            ):
                file_path = str(request.url.path)[len("/filebrowser/api/resources") :]
                index_name = str(request.url.path).split("/")[4]
                with batch_lock:
                    if index_name in batch_files:
                        batch_files[index_name].append(
                            f"{DEFAULT_KNOWLEDGE_PATH}{file_path}"
                        )
                    else:
                        batch_files[index_name] = [
                            f"{DEFAULT_KNOWLEDGE_PATH}{file_path}"
                        ]
                logger.info(
                    f"File added to processor: {index_name}, {DEFAULT_KNOWLEDGE_PATH}{file_path}"
                )
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
    else:
        response = await call_next(request)
        return response
