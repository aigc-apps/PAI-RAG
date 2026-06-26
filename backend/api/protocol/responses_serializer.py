from __future__ import annotations
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from agent.core.events import (
    TextDelta,
    ReasoningDelta,
    ToolStarted,
    ToolCompleted,
    ToolResult,
    RunCompleted,
    RunFailed,
)
from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseFunctionToolCall,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import Content
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from openai.types.responses.response_error import ResponseError
from openai.types.responses import (
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseReasoningTextDeltaEvent,
)


def _usage(u) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=u.input,
        output_tokens=u.output,
        total_tokens=u.total,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


class _Assembler:
    """Consumes an AgentEvent stream into OpenAI Response output items + store items.
    Shared by the sync and streaming serializers."""

    def __init__(
        self, model: str, response_id: str, conversation_id: Optional[str]
    ):
        self.model = model
        self.response_id = response_id
        self.conversation_id = conversation_id
        self.created_at = time.time()
        self.output: List[Any] = []  # OpenAI output items
        self.store_items: List[
            Dict
        ] = []  # store-ready dicts {type, role, content}
        self.text = ""
        self.reasoning = ""
        self.usage = None
        self.status = "completed"
        self.error: Optional[ResponseError] = None

    def _msg_item_id(self) -> str:
        return f"msg_{self.response_id}"

    def on_text(self, text: str):
        self.text += text

    def on_reasoning(self, text: str):
        self.reasoning += text

    def on_tool_completed(self, call_id: str, name: str, arguments: str):
        self.output.append(
            ResponseFunctionToolCall(
                id=f"fc_{call_id}",
                call_id=call_id,
                name=name,
                arguments=arguments or "",
                type="function_call",
                status="completed",
            )
        )
        self.store_items.append(
            {
                "type": "function_call",
                "role": None,
                "content": {
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments or "",
                },
            }
        )

    def on_tool_result(
        self, call_id: str, output: Optional[str], error: Optional[str]
    ):
        self.store_items.append(
            {
                "type": "function_call_output",
                "role": None,
                "content": {
                    "call_id": call_id,
                    "output": output if output is not None else (error or ""),
                },
            }
        )

    def on_failed(self, message: str):
        self.status = "failed"
        self.error = ResponseError(code="server_error", message=message)

    def finalize(self, usage) -> None:
        # reasoning item (if any) first, then the assistant message
        if self.reasoning:
            self.output.insert(
                0,
                ResponseReasoningItem(
                    id=f"rs_{self.response_id}",
                    type="reasoning",
                    status="completed",
                    summary=[],
                    content=[
                        Content(text=self.reasoning, type="reasoning_text")
                    ],
                ),
            )
            self.store_items.append(
                {
                    "type": "reasoning",
                    "role": None,
                    "content": {"text": self.reasoning},
                }
            )
        if self.text or self.status == "completed":
            self.output.append(
                ResponseOutputMessage(
                    id=self._msg_item_id(),
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        ResponseOutputText(
                            annotations=[], text=self.text, type="output_text"
                        )
                    ],
                )
            )
            self.store_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": {"text": self.text},
                }
            )
        if usage is not None:
            self.usage = _usage(usage)

    def to_response(self) -> Response:
        return Response(
            id=self.response_id,
            created_at=self.created_at,
            model=self.model,
            object="response",
            output=self.output,
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            status=self.status,
            usage=self.usage,
            error=self.error,
            previous_response_id=None,
            conversation=(
                {"id": self.conversation_id} if self.conversation_id else None
            ),
        )


async def serialize_response_sync(
    events: AsyncIterator,
    *,
    model: str,
    response_id: str,
    conversation_id: Optional[str],
) -> Tuple[dict, List[Dict]]:
    """Consume the full AgentEvent stream and return (response_dict, store_items).
    response_dict is `Response.model_dump()` (JSON-ready, OpenAI-conformant).
    """
    asm = _Assembler(model, response_id, conversation_id)
    usage = None
    async for ev in events:
        if isinstance(ev, TextDelta):
            asm.on_text(ev.text)
        elif isinstance(ev, ReasoningDelta):
            asm.on_reasoning(ev.text)
        elif isinstance(ev, ToolCompleted):
            asm.on_tool_completed(ev.call_id, ev.name, ev.arguments)
        elif isinstance(ev, ToolResult):
            asm.on_tool_result(ev.call_id, ev.output, ev.error)
        elif isinstance(ev, RunCompleted):
            usage = ev.usage
        elif isinstance(ev, RunFailed):
            asm.on_failed(ev.message)
        # RunStarted, ToolStarted: no sync effect
    asm.finalize(usage)
    return asm.to_response().model_dump(mode="json"), asm.store_items


def _sse(event) -> str:
    return f"data: {event.model_dump_json()}\n\n"


async def serialize_response_stream(
    events: AsyncIterator,
    *,
    model: str,
    response_id: str,
    conversation_id: Optional[str],
    sink: Dict,
) -> AsyncIterator[str]:
    """AgentEvent stream -> SSE `response.*` event strings (OpenAI order). On completion,
    sink["response"] = final Response.model_dump(mode="json") and sink["items"] = store items.
    """
    asm = _Assembler(model, response_id, conversation_id)
    seq = 0

    def nxt() -> int:
        nonlocal seq
        seq += 1
        return seq

    next_index = 0

    def alloc_index() -> int:
        nonlocal next_index
        idx = next_index
        next_index += 1
        return idx

    # response.created + response.in_progress
    yield _sse(
        ResponseCreatedEvent(
            response=asm.to_response(),
            sequence_number=nxt(),
            type="response.created",
        )
    )
    yield _sse(
        ResponseInProgressEvent(
            response=asm.to_response(),
            sequence_number=nxt(),
            type="response.in_progress",
        )
    )

    msg_open = False
    msg_index = 0
    msg_item_id = asm._msg_item_id()
    tool_indices: Dict[str, int] = {}  # call_id -> output_index
    reasoning_index = None
    usage = None

    async for ev in events:
        if isinstance(ev, TextDelta):
            if not msg_open:
                # open a message output item + a text content part
                msg_index = alloc_index()
                placeholder = ResponseOutputMessage(
                    id=msg_item_id,
                    role="assistant",
                    status="in_progress",
                    type="message",
                    content=[],
                )
                yield _sse(
                    ResponseOutputItemAddedEvent(
                        item=placeholder,
                        output_index=msg_index,
                        sequence_number=nxt(),
                        type="response.output_item.added",
                    )
                )
                yield _sse(
                    ResponseContentPartAddedEvent(
                        content_index=0,
                        item_id=msg_item_id,
                        output_index=msg_index,
                        part=ResponseOutputText(
                            annotations=[], text="", type="output_text"
                        ),
                        sequence_number=nxt(),
                        type="response.content_part.added",
                    )
                )
                msg_open = True
            asm.on_text(ev.text)
            yield _sse(
                ResponseTextDeltaEvent(
                    content_index=0,
                    delta=ev.text,
                    item_id=msg_item_id,
                    logprobs=[],
                    output_index=msg_index,
                    sequence_number=nxt(),
                    type="response.output_text.delta",
                )
            )
        elif isinstance(ev, ReasoningDelta):
            asm.on_reasoning(ev.text)
            # surface reasoning text as a streaming delta (item assembled at finalize)
            if reasoning_index is None:
                reasoning_index = alloc_index()
            # TODO: emit reasoning output_item.added/done for full conformance
            yield _sse(
                ResponseReasoningTextDeltaEvent(
                    content_index=0,
                    delta=ev.text,
                    item_id=f"rs_{response_id}",
                    output_index=reasoning_index,
                    sequence_number=nxt(),
                    type="response.reasoning_text.delta",
                )
            )
        elif isinstance(ev, ToolStarted):
            idx = alloc_index()
            tool_indices[ev.call_id] = idx
            yield _sse(
                ResponseOutputItemAddedEvent(
                    item=ResponseFunctionToolCall(
                        id=f"fc_{ev.call_id}",
                        call_id=ev.call_id,
                        name=ev.name,
                        arguments="",
                        type="function_call",
                        status="in_progress",
                    ),
                    output_index=idx,
                    sequence_number=nxt(),
                    type="response.output_item.added",
                )
            )
        elif isinstance(ev, ToolCompleted):
            asm.on_tool_completed(ev.call_id, ev.name, ev.arguments)
            idx = tool_indices.get(ev.call_id)
            if idx is None:
                # No ToolStarted arrived: open the item now.
                idx = alloc_index()
                tool_indices[ev.call_id] = idx
                yield _sse(
                    ResponseOutputItemAddedEvent(
                        item=ResponseFunctionToolCall(
                            id=f"fc_{ev.call_id}",
                            call_id=ev.call_id,
                            name=ev.name,
                            arguments="",
                            type="function_call",
                            status="in_progress",
                        ),
                        output_index=idx,
                        sequence_number=nxt(),
                        type="response.output_item.added",
                    )
                )
            yield _sse(
                ResponseFunctionCallArgumentsDeltaEvent(
                    delta=ev.arguments or "",
                    item_id=f"fc_{ev.call_id}",
                    output_index=idx,
                    sequence_number=nxt(),
                    type="response.function_call_arguments.delta",
                )
            )
            yield _sse(
                ResponseFunctionCallArgumentsDoneEvent(
                    arguments=ev.arguments or "",
                    item_id=f"fc_{ev.call_id}",
                    name=ev.name,
                    output_index=idx,
                    sequence_number=nxt(),
                    type="response.function_call_arguments.done",
                )
            )
            yield _sse(
                ResponseOutputItemDoneEvent(
                    item=ResponseFunctionToolCall(
                        id=f"fc_{ev.call_id}",
                        call_id=ev.call_id,
                        name=ev.name,
                        arguments=ev.arguments or "",
                        type="function_call",
                        status="completed",
                    ),
                    output_index=idx,
                    sequence_number=nxt(),
                    type="response.output_item.done",
                )
            )
        elif isinstance(ev, ToolResult):
            asm.on_tool_result(ev.call_id, ev.output, ev.error)
        elif isinstance(ev, RunCompleted):
            usage = ev.usage
        elif isinstance(ev, RunFailed):
            asm.on_failed(ev.message)

    # close an open message item
    if msg_open:
        yield _sse(
            ResponseTextDoneEvent(
                content_index=0,
                item_id=msg_item_id,
                logprobs=[],
                output_index=msg_index,
                sequence_number=nxt(),
                text=asm.text,
                type="response.output_text.done",
            )
        )
        yield _sse(
            ResponseContentPartDoneEvent(
                content_index=0,
                item_id=msg_item_id,
                output_index=msg_index,
                part=ResponseOutputText(
                    annotations=[], text=asm.text, type="output_text"
                ),
                sequence_number=nxt(),
                type="response.content_part.done",
            )
        )

    asm.finalize(usage)
    if msg_open:
        yield _sse(
            ResponseOutputItemDoneEvent(
                item=ResponseOutputMessage(
                    id=msg_item_id,
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        ResponseOutputText(
                            annotations=[], text=asm.text, type="output_text"
                        )
                    ],
                ),
                output_index=msg_index,
                sequence_number=nxt(),
                type="response.output_item.done",
            )
        )

    final = asm.to_response()
    if asm.status == "failed":
        yield _sse(
            ResponseFailedEvent(
                response=final, sequence_number=nxt(), type="response.failed"
            )
        )
    else:
        yield _sse(
            ResponseCompletedEvent(
                response=final,
                sequence_number=nxt(),
                type="response.completed",
            )
        )

    sink["response"] = final.model_dump(mode="json")
    sink["items"] = asm.store_items
