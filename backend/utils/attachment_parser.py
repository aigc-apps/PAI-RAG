import json
from typing import List
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed
from llama_index.core.tools.function_tool import ToolOutput
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from chat.tools.attachments.file_reader import aget_file_reader
from chat.tools.attachments.image_parser import aget_image_parser_tool
from chat.llm.models import ToolResultChunk, TextChunk

from loguru import logger


class AttachmentInputData(BaseModel):
    messages: List[dict] = []
    chunks: List[TextChunk] = []


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(async_fn, fn_args) -> ToolOutput:
    return await async_fn.acall(**fn_args)


async def parse_attachments_from_messages(messages: List[dict], question: str = ""):
    tool_call_chunks = []
    last_user_message = messages[-1]
    if last_user_message.get("role") == "user":
        user_attachments = last_user_message.get("attachments", [])
        if len(user_attachments) > 0:
            for attachment in user_attachments:
                if str(attachment.get("contentType")).startswith("image/"):
                    # for image attachments
                    image_parser = await aget_image_parser_tool()
                    image_parser_fn_args = {
                        "file_id": attachment.get("id"),
                        "question": question,
                    }
                    image_parser_tool_call = ChoiceDeltaToolCall(
                        index=0,
                        id=f"call_image_parser_{attachment.get('id')}",
                        type="function",
                        function=ChoiceDeltaToolCallFunction(
                            name=image_parser.metadata.name,
                            arguments=json.dumps(
                                image_parser_fn_args, ensure_ascii=False
                            ),
                        ),
                    )
                    logger.info(
                        f"Calling tool [image_parser] with args {image_parser_fn_args}."
                    )
                    tool_result = await call_tool_with_retry(
                        image_parser, image_parser_fn_args
                    )
                    logger.info(f"Get tool result {tool_result}.")
                    reply_text = "\n\n 以下是针对图片附件的分析结果："
                    try:
                        result_data = json.loads(tool_result.content)
                        if "error" in result_data:
                            reply_text += f"❌ 图片解析失败：{result_data['error']}"
                        else:
                            question = result_data.get("question") or "未指定问题"
                            answer = result_data.get("answer", "无返回内容")
                            file_id = result_data.get("file_id", "未知文件")
                            reply_text += answer
                    except (json.JSONDecodeError, TypeError):
                        reply_text += tool_result.content

                    # 只在user message最后追加文件读取结果，不使用 tool_call / tool 消息
                    last_user_message["content"][0]["text"] += reply_text
                    tool_call_chunks.append(
                        TextChunk(
                            tool_calls=[image_parser_tool_call],
                        )
                    )
                    tool_call_chunks.append(
                        ToolResultChunk(
                            tool=image_parser_tool_call,
                            result=tool_result.content,
                        )
                    )
                else:
                    # for text attachments
                    file_reader = await aget_file_reader()
                    file_id = attachment.get("id")
                    file_name = attachment.get("name", "未知附件")
                    file_reader_fn_args = {
                        "file_id": file_id,
                        "file_name": file_name,
                    }
                    file_reader_tool_call = ChoiceDeltaToolCall(
                        index=0,
                        id=f"call_file_reader_{file_id}",
                        type="function",
                        function=ChoiceDeltaToolCallFunction(
                            name=file_reader.metadata.name,
                            arguments=json.dumps(
                                file_reader_fn_args, ensure_ascii=False
                            ),
                        ),
                    )
                    logger.info(
                        f"Calling tool [file_reader] with args {file_reader_fn_args}."
                    )
                    tool_result = await call_tool_with_retry(
                        file_reader, file_reader_fn_args
                    )
                    logger.info(f"Get tool result {tool_result}.")
                    reply_text = "\n\n 以下是附件的解析结果："
                    try:
                        result_data = json.loads(tool_result.content)
                        reply_text = f"📄 文件“{file_name}” (ID:{file_id}) 的内容如下：\n\n {result_data.get('data', '无内容')}"
                    except (json.JSONDecodeError, TypeError):
                        reply_text = f"📄 文件“{file_name}” (ID:{file_id}) 的内容如下：\n\n {tool_result.content}"

                    # 只在user message最后追加文件读取结果，不使用 tool_call / tool 消息
                    last_user_message["content"][0]["text"] += reply_text
                    tool_call_chunks.append(
                        TextChunk(
                            tool_calls=[file_reader_tool_call],
                        )
                    )
                    tool_call_chunks.append(
                        ToolResultChunk(
                            tool=file_reader_tool_call,
                            result=tool_result.content,
                        )
                    )

        messages[-1] = last_user_message

    return AttachmentInputData(
        messages=messages, chunks=tool_call_chunks
    )
