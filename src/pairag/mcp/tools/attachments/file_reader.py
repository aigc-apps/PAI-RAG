import json
from llama_index.core.tools import FunctionTool

ATTACHMENTS_DIR = "localdata/attachments"


async def aget_file_content(file_id: str, file_name: str = None):
    """Get read file tool"""
    with open(f"{ATTACHMENTS_DIR}/{file_id}.txt", "r", encoding="utf-8") as f:
        content = f.read()

    return json.dumps({"data": content}, ensure_ascii=False)


async def aget_file_reader():
    read_file_tool = FunctionTool.from_defaults(
        async_fn=aget_file_content,
        name="read-file",
        description="根据提供的附件ID读取文件的内容。",
    )
    return read_file_tool
