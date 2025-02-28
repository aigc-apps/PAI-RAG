from fastapi import Request, Response
from pai_rag.core.rag_knowledgebase_manager import DEFAULT_KNOWLEDGE_PATH
from pai_rag.app.web.rag_local_client import rag_client
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


def process_delete_mothod(url_path: str):
    path_parts = url_path.strip("/").split("/")
    if len(path_parts) == 4 and url_path.endswith("/"):
        index_name = path_parts[3]
        rag_client.delete_index(index_name)
        logger.info(f"Index Deleted. Index name: {index_name}")
    elif len(path_parts) >= 6 and path_parts[4] == "docs":
        if not url_path.endswith("/"):
            index_name = path_parts[3]
            file_path = url_path[len("/filebrowser/api/resources") :]
            real_file_path = f"{DEFAULT_KNOWLEDGE_PATH}{file_path}"
            is_del = rag_client.delete_file_from_index(index_name, real_file_path)
            if is_del:
                logger.info(
                    f"File Deleted. File name: {real_file_path} from index {index_name}"
                )
            else:
                raise IndexError(
                    "Delete operation is not supported for the given index type."
                )
    else:
        logger.warning(f"Invalid path format: insufficient path parts {url_path}")
        raise IndexError(f"Invalid path format {url_path}.")


def process_post_method(url_path: str):
    if not url_path.endswith("/"):
        path_parts = url_path.strip("/").split("/")
        if len(path_parts) >= 6 and path_parts[4] == "docs":
            file_path = url_path[len("/filebrowser/api/resources") :]
            real_file_path = f"{DEFAULT_KNOWLEDGE_PATH}{file_path}"
            index_name = path_parts[3]
            rag_client.add_file_to_index(index_name, real_file_path)
        else:
            logger.warning(f"Invalid path format: insufficient path parts {url_path}")
            raise ValueError(f"Invalid path format {url_path}.")
    else:
        logger.info(f"skip directory {url_path}")


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
                process_delete_mothod(str(request.url.path))
            elif (
                "/filebrowser/api/resources" in request.url.path
                and request.method.lower() == "post"
            ):
                process_post_method(str(request.url.path))

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
