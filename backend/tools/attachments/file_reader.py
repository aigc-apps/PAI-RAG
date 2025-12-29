import json
from typing import Annotated
from llama_index.core.tools import FunctionTool

from service.knowledgebase.file_service import FileService



async def aget_file_reader(file_service: FileService = None, tenant_id: str = None):
    if not file_service:
        raise ValueError("file_service is required")

    async def aread_file_content(
        file_id: Annotated[
            str,
            "要读取的文件的ID，必须提供。",
        ] = None,
        **kwargs
    ):
        # 从 kwargs 中获取 file_id（如果作为关键字参数传递）
        if file_id is None:
            file_id = kwargs.get("file_id")

        if not file_id:
            error_msg = "缺少必需参数 file_id。请提供要读取的文件ID。"
            return json.dumps({"error": error_msg}, ensure_ascii=False)

        file_entity = await file_service.get_file_by_id(file_id=file_id, tenant_id=tenant_id)
        if not file_entity:
            raise ValueError(f"File entity not found for attachment {file_id}")

        content = file_entity.file_content or ""
        file_name = file_entity.file_name or ""
        result = f"📄 文件“{file_name}” (文件ID:{file_id}) 的内容如下：\n\n {content}"
        if file_entity.file_content_length > 1000:
            result += "\n\n[The file content is too long, has been truncated]"
        return json.dumps({"data": result}, ensure_ascii=False)

    read_file_tool = FunctionTool.from_defaults(
        async_fn=aread_file_content,
        name="read-file",
        description="根据提供的附件ID读取文件的内容。\n参数：\n- file_id (str, 必需): 要读取的文件的ID。",
    )
    return read_file_tool
