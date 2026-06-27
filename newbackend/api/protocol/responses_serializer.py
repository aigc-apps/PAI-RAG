from __future__ import annotations
import asyncio
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
from openai.types.responses.response_reasoning_item import Summary
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
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningSummaryPartDoneEvent,
)
from openai.types.responses.response_reasoning_summary_part_added_event import (
    Part as SummaryPartAdded,
)
from openai.types.responses.response_reasoning_summary_part_done_event import (
    Part as SummaryPartDone,
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
                    summary=[
                        Summary(text=self.reasoning, type="summary_text")
                    ],
                    content=[],
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


import json as _json


def _sse_obj(obj: dict) -> str:
    return f"data: {_json.dumps(obj, ensure_ascii=False)}\n\n"


class _Cancelled(Exception):
    """Raised inside the stream loop when the cancel event fires."""


async def _anext_or_cancel(aiter, cancel):
    """Return the next event, or raise _Cancelled if the cancel event fires first.
    Without a cancel event this is a plain anext."""
    if cancel is None:
        return await aiter.__anext__()
    next_task = asyncio.ensure_future(aiter.__anext__())
    cancel_task = asyncio.ensure_future(cancel.wait())
    try:
        done, _pending = await asyncio.wait(
            {next_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
    except BaseException:
        next_task.cancel()
        cancel_task.cancel()
        raise
    if next_task in done:
        cancel_task.cancel()
        return next_task.result()  # may raise StopAsyncIteration
    next_task.cancel()
    raise _Cancelled()


async def serialize_response_stream(
    events: AsyncIterator,
    *,
    model: str,
    response_id: str,
    conversation_id: Optional[str],
    sink: Dict,
    cancel: Optional[asyncio.Event] = None,
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
    reasoning_open = False
    rs_id = f"rs_{response_id}"
    usage = None

    def close_reasoning():
        """Emit the closing reasoning-summary envelope (text.done, part.done,
        output_item.done). Yields SSE strings; caller must `yield from`."""
        full = asm.reasoning
        yield _sse(
            ResponseReasoningSummaryTextDoneEvent(
                item_id=rs_id,
                output_index=reasoning_index,
                summary_index=0,
                text=full,
                sequence_number=nxt(),
                type="response.reasoning_summary_text.done",
            )
        )
        yield _sse(
            ResponseReasoningSummaryPartDoneEvent(
                item_id=rs_id,
                output_index=reasoning_index,
                summary_index=0,
                part=SummaryPartDone(text=full, type="summary_text"),
                sequence_number=nxt(),
                type="response.reasoning_summary_part.done",
            )
        )
        yield _sse(
            ResponseOutputItemDoneEvent(
                item=ResponseReasoningItem(
                    id=rs_id,
                    type="reasoning",
                    status="completed",
                    summary=[Summary(text=full, type="summary_text")],
                    content=[],
                ),
                output_index=reasoning_index,
                sequence_number=nxt(),
                type="response.output_item.done",
            )
        )

    cancelled = False
    aiter = events.__aiter__()
    while True:
        try:
            ev = await _anext_or_cancel(aiter, cancel)
        except StopAsyncIteration:
            break
        except _Cancelled:
            cancelled = True
            break
        if isinstance(ev, TextDelta):
            if reasoning_open:
                for chunk in close_reasoning():
                    yield chunk
                reasoning_open = False
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
            if not reasoning_open:
                # open the reasoning output item + summary part (comes first)
                reasoning_index = alloc_index()
                reasoning_open = True
                yield _sse(
                    ResponseOutputItemAddedEvent(
                        item=ResponseReasoningItem(
                            id=rs_id,
                            type="reasoning",
                            status="in_progress",
                            summary=[],
                            content=[],
                        ),
                        output_index=reasoning_index,
                        sequence_number=nxt(),
                        type="response.output_item.added",
                    )
                )
                yield _sse(
                    ResponseReasoningSummaryPartAddedEvent(
                        item_id=rs_id,
                        output_index=reasoning_index,
                        summary_index=0,
                        part=SummaryPartAdded(text="", type="summary_text"),
                        sequence_number=nxt(),
                        type="response.reasoning_summary_part.added",
                    )
                )
            asm.on_reasoning(ev.text)
            yield _sse(
                ResponseReasoningSummaryTextDeltaEvent(
                    delta=ev.text,
                    item_id=rs_id,
                    output_index=reasoning_index,
                    summary_index=0,
                    sequence_number=nxt(),
                    type="response.reasoning_summary_text.delta",
                )
            )
        elif isinstance(ev, ToolStarted):
            if reasoning_open:
                for chunk in close_reasoning():
                    yield chunk
                reasoning_open = False
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
            if reasoning_open:
                for chunk in close_reasoning():
                    yield chunk
                reasoning_open = False
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
            yield _sse_obj({
                "type": "response.tool_result",
                "call_id": ev.call_id,
                "output": ev.output if ev.output is not None else (ev.error or ""),
                "ok": ev.ok,
                "sequence_number": nxt(),
            })
        elif isinstance(ev, RunCompleted):
            usage = ev.usage
        elif isinstance(ev, RunFailed):
            asm.on_failed(ev.message)

    # close an open reasoning item (reasoning-only run: no text/tool ever arrived)
    if reasoning_open:
        for chunk in close_reasoning():
            yield chunk
        reasoning_open = False

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

    if cancelled:
        asm.status = "cancelled"

    final = asm.to_response()
    if asm.status == "failed":
        yield _sse(
            ResponseFailedEvent(
                response=final, sequence_number=nxt(), type="response.failed"
            )
        )
    elif asm.status == "cancelled":
        yield _sse(
            ResponseIncompleteEvent(
                response=final, sequence_number=nxt(), type="response.incomplete"
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
