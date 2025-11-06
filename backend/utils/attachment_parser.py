import json
from typing import Any, List
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
from config.providers.code_sandbox_provider import codesandbox_provider
from rag.chunk_helper import read_file_from_db
from chat.tools.code_sandbox_tool import DEFAULT_CODE_SANDBOX_DIR_PATH
import os
from loguru import logger


class AttachmentInputData(BaseModel):
    messages: List[dict] = []
    chunks: List[TextChunk] = []


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(async_fn, fn_args) -> ToolOutput:
    from extensions.trace.pai_agent_wrapper import instrument_async_call
    return await instrument_async_call(async_fn, fn_args)



def append_text(user_message: Any, text: str):
    assert "content" in user_message, "Message必须包含content字段"

    if isinstance(user_message["content"], str):
        user_message["content"] += text
    else:
        for block in user_message["content"]:
            if block.get("type") == "text":
                block["text"] += text
                return


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
                    append_text(last_user_message, reply_text)
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
                    file_reader_fn_args = {
                        "file_id": file_id
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
                        reply_text = result_data.get('data', '无内容')
                    except (json.JSONDecodeError, TypeError):
                        reply_text = tool_result.content

                    # 只在user message最后追加文件读取结果，不使用 tool_call / tool 消息
                    append_text(last_user_message, reply_text)
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

    # 列举code sandbox里的文件
    user_attachments = []
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            attachments_in_message = message.get("attachments", [])
            attachment_names_in_message = []
            if len(attachments_in_message) > 0:
                for attachment in attachments_in_message:
                    attachment_file_entity = await read_file_from_db(file_id=attachment.get("id"))
                    name = attachment_file_entity.file_name
                    if not name:
                        logger.warning("Attachment missing 'name' field, skipping: %s", attachment)
                        continue
                    attachment_names_in_message.append(name)
                if codesandbox_provider.tool and codesandbox_provider.tool.enabled:
                    # 只在user message最后追加列出文件结果，不使用 tool_call / tool 消息
                    attachment_names_in_message = [os.path.join(DEFAULT_CODE_SANDBOX_DIR_PATH, attachment_name) for attachment_name in attachment_names_in_message]
                    attachment_names_in_message = ','.join(attachment_names_in_message)
                    reply_text = f"\n\n 可以参考以下文件路径回答用户问题：\n\n {attachment_names_in_message}"
                    append_text(message, reply_text)

    return AttachmentInputData(
        messages=messages, chunks=tool_call_chunks
    )
