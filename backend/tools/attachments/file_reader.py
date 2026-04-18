"""Agent tool: read the extracted text of an attached file.

Registration is cheap — only the name → id mapping is captured in closure.
Content is fetched at each tool call so extraction that completes after
`parse_attachment_tools` has already run is still visible. If extraction
is still in flight at call time the tool briefly polls before giving up
with a clear "still processing" message rather than lying "file is empty".
"""
import asyncio
import json
from typing import Annotated, List

from llama_index.core.tools import FunctionTool
from loguru import logger

from common.knowledgebase.types import FileStatus
from db.db_context import create_db_session
from service.file.file_resource_service import FileResourceService


# Poll the extraction state for up to this long when the file is still
# parsing at tool-call time. Covers the common race where the user sends
# a chat message before the worker has finished processing a freshly-uploaded
# attachment. Longer than typical PDF extraction (~1–3s on a few-MB file) but
# short enough that a broken/stuck extraction doesn't hang the agent.
POLL_TIMEOUT_SECONDS = 15.0
POLL_INTERVAL_SECONDS = 0.5

# Mirror of FileResourceService.LLM_INLINE_TEXT_LIMIT — trims what the tool
# returns into the LLM context so a 500KB extract doesn't blow the window.
INLINE_TEXT_LIMIT = FileResourceService.LLM_INLINE_TEXT_LIMIT


async def aget_file_reader(file_ids: List[str], tenant_id: str):
    if not file_ids:
        raise ValueError("file_ids is required")

    # Name → id mapping is stable across the chat turn; pre-fetch so the
    # tool doesn't have to.
    async with create_db_session() as session:
        svc = FileResourceService(session)
        files = await svc.get_files(file_ids=list(file_ids), tenant_id=tenant_id)
    name_to_id = {f.file_name: f.id for f in files if f.file_name}

    async def aread_file_content(
        file_name: Annotated[
            str,
            "要读取的文件的名称，必须提供。",
        ] = None,
    ):
        if not file_name:
            raise ValueError("file_name is required")

        file_id = name_to_id.get(file_name)
        if file_id is None:
            return json.dumps({
                "error": f"未找到文件 '{file_name}'",
                "available": list(name_to_id.keys()),
            }, ensure_ascii=False)

        deadline = asyncio.get_event_loop().time() + POLL_TIMEOUT_SECONDS
        while True:
            async with create_db_session() as session:
                svc = FileResourceService(session)
                entity = await svc.get_file(file_id=file_id, tenant_id=tenant_id)
                if entity is None:
                    return json.dumps(
                        {"error": f"文件 '{file_name}' 不存在"},
                        ensure_ascii=False,
                    )

                if entity.status == FileStatus.failed.value:
                    return json.dumps({
                        "error": f"文件 '{file_name}' 解析失败: {entity.failed_reason or '未知原因'}",
                    }, ensure_ascii=False)

                row = await svc.get_text_content(
                    file_id=file_id, tenant_id=tenant_id
                )
                if row is not None and row.content:
                    total = row.content_length or len(row.content)
                    inline = row.content[:INLINE_TEXT_LIMIT]
                    trailer = ""
                    if total > INLINE_TEXT_LIMIT:
                        trailer = (
                            f"\n\n[已截断：显示 {len(inline)} / {total} 字符，"
                            f"可通过 `search-file-chunks` 工具按关键字检索更多。]"
                        )
                    body = f"📄 文件“{file_name}” 的内容如下：\n\n{inline}{trailer}"
                    return json.dumps({"data": body}, ensure_ascii=False)

                # No content yet. If extraction has already reported a
                # terminal success, this is a truly empty / non-text file —
                # return that honestly instead of looping forever.
                if entity.status in (
                    FileStatus.succeeded.value,
                    FileStatus.cancelled.value,
                ):
                    return json.dumps({
                        "data": f"📄 文件“{file_name}” 没有可提取的文本内容。",
                    }, ensure_ascii=False)

            # Still pending/parsing/persisting — wait a bit and retry.
            if asyncio.get_event_loop().time() >= deadline:
                logger.warning(
                    f"[file_reader] timed out waiting for {file_id} ({file_name}) "
                    f"after {POLL_TIMEOUT_SECONDS}s; status={entity.status}"
                )
                return json.dumps({
                    "warning": (
                        f"文件 '{file_name}' 仍在解析中（已等待 "
                        f"{POLL_TIMEOUT_SECONDS:.0f} 秒），请稍后重试。"
                    ),
                }, ensure_ascii=False)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    return FunctionTool.from_defaults(
        async_fn=aread_file_content,
        name="read-file",
        description=(
            "根据提供的文件名称读取文件的内容。\n"
            "参数：\n"
            "- file_name (str, 必需): 要读取的文件的名称。"
        ),
        return_direct=False,
    )
