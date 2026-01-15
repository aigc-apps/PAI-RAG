import json
from typing import Annotated, Dict
from llama_index.core.tools import FunctionTool

async def aget_file_reader(file_contents_map: Dict[str, str]):
    if not file_contents_map:
        raise ValueError("file_contents_map is required")

    async def aread_file_content(
        file_name: Annotated[
            str,
            "要读取的文件的名称，必须提供。",
        ] = None,
    ):
        if not file_name:
            raise ValueError("file_name is required")

        content = file_contents_map.get(file_name)
        if not content:
            raise ValueError(f"File content not found for attachment {file_name}")

        return json.dumps({"data": content}, ensure_ascii=False)

    read_file_tool = FunctionTool.from_defaults(
        async_fn=aread_file_content,
        name="read-file",
        description="根据提供的文件名称读取文件的内容。\n参数：\n- file_name (str, 必需): 要读取的文件的名称。",
        return_direct=False,
    )
    return read_file_tool
