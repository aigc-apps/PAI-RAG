import json
from typing import List
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

class ReturnDirectConfig:
    def __init__(self):
        self._return_direct = None
        self._content = ""

    @property
    def return_direct(self):
        return self._return_direct

    @return_direct.setter
    def return_direct(self, value):
        if value not in (None, True, False):
            logger.info("return_direct 只能设置为 None, True 或 False")
            return

        current = self._return_direct

        # 规则：设为 False 后不可更改
        if current is False:
            logger.info("return_direct 已被设为 False，不能再设为其他值")
            return

        self._return_direct = value

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def call_tool_with_retry(async_fn, fn_args) -> ToolOutput:
    return await async_fn.acall(**fn_args)


async def parse_attchments_from_messages(messages: List[dict], question: str = ""):
    ret_messages = messages
    tool_call_chunks = []
    cfg = ReturnDirectConfig()
    attachments = []
    for message in messages:
        if message.get("role") == "user":
            user_attachments = message.get("attachments", [])
            if len(user_attachments) > 0:
                attachments.extend(user_attachments)
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
                        logger.info(f"Calling tool [image_parser] with args {image_parser_fn_args}.")
                        tool_result = await call_tool_with_retry(image_parser, image_parser_fn_args)
                        logger.info(f"Get tool result {tool_result}.")
                        cfg.return_direct = json.loads(tool_result.content).get("return_direct", False)
                        cfg._content = json.loads(tool_result.content).get("answer", "")
                        ret_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    image_parser_tool_call
                                ]
                            }
                        )
                        ret_messages.append(
                            {
                                "role": "tool",
                                "content": tool_result.content,
                                "tool_call_id": image_parser_tool_call.id
                            }
                        )
                        tool_call_chunks.append(TextChunk(
                            tool_calls=[image_parser_tool_call],
                        ))
                        tool_call_chunks.append(ToolResultChunk(
                            tool=image_parser_tool_call,
                            result=tool_result.content,
                        ))
                    else:
                        # for text attachments
                        file_reader = await aget_file_reader()
                        file_reader_fn_args = {
                            "file_id": attachment.get("id"),
                            "file_name": attachment.get("name", "未知附件"),
                        }
                        file_reader_tool_call = ChoiceDeltaToolCall(
                            index=0,
                            id=f"call_file_reader_{attachment.get('id')}",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name=file_reader.metadata.name,
                                arguments=json.dumps(
                                    file_reader_fn_args, ensure_ascii=False
                                ),
                            ),
                        )
                        logger.info(f"Calling tool [file_reader] with args {file_reader_fn_args}.")
                        tool_result = await call_tool_with_retry(file_reader, file_reader_fn_args)
                        logger.info(f"Get tool result {tool_result}.")
                        cfg.return_direct = False
                        ret_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    file_reader_tool_call
                                ]
                            }
                        )
                        ret_messages.append(
                            {
                                "role": "tool",
                                "content": tool_result.content,
                                "tool_call_id": file_reader_tool_call.id
                            }
                        )
                        tool_call_chunks.append(TextChunk(
                            tool_calls=[file_reader_tool_call],
                        ))
                        tool_call_chunks.append(ToolResultChunk(
                            tool=file_reader_tool_call,
                            result=tool_result.content,
                        ))
    if len(attachments) > 1:
        cfg.return_direct = False
    return ret_messages, tool_call_chunks, cfg.return_direct, cfg._content
