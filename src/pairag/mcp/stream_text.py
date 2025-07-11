from typing import List
import json
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from llama_index.core.base.llms.types import (
    MessageRole,
    ChatResponseAsyncGen,
)
from loguru import logger


# data stream writer for bridging vercel ai and llamaindex, see https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
class VercelAiDataStreamWriter:
    async def astream_text(self, async_response_gen: ChatResponseAsyncGen):
        logger.info("Start generating chunks.")
        async for response in async_response_gen:
            if response.message.role == MessageRole.ASSISTANT:
                is_error_message = response.message.additional_kwargs.get(
                    "failed", False
                )
                if is_error_message:
                    yield f"3:{json.dumps(response.delta, ensure_ascii=False)}\n"
                    continue

                tool_calls: List[
                    ChoiceDeltaToolCall
                ] = response.message.additional_kwargs.get("tool_calls", [])
                if response.delta:
                    # text stream
                    yield f"0:{json.dumps(response.delta, ensure_ascii=False)}\n"
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_call_text = json.dumps(
                            {
                                "toolCallId": tool_call.id,
                                "toolName": tool_call.function.name,
                                "args": json.loads(tool_call.function.arguments),
                            },
                            ensure_ascii=False,
                        )
                        yield f"9:{tool_call_text}\n"

            elif response.message.role == MessageRole.TOOL:
                # tool results
                tool_call_id = response.message.additional_kwargs["tool_call_id"]
                tool_result_text = json.dumps(
                    {
                        "toolCallId": tool_call_id,
                        "result": response.delta,
                    },
                    ensure_ascii=False,
                )
                yield f"a:{tool_result_text}\n"
            else:
                raise ValueError(f"Unknown role: {response.message.role}")

        logger.info("Finished generating chunks.")


class AgentFinalAnswerWriter:
    async def astream_text(self, async_response_gen: ChatResponseAsyncGen):
        logger.info("Start generating final answer.")
        final_answer = ""
        async for response in async_response_gen:
            if response.message.role == MessageRole.ASSISTANT:
                is_error_message = response.message.additional_kwargs.get(
                    "failed", False
                )
                if is_error_message:
                    continue

                tool_calls: List[
                    ChoiceDeltaToolCall
                ] = response.message.additional_kwargs.get("tool_calls", [])
                if response.delta and not tool_calls:
                    final_answer += response.delta

        return final_answer
