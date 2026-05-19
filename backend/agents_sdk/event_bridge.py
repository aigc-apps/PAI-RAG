"""Map SDK stream events to public API wire chunks and audit rows.

Output channels per event:

- :func:`to_responses_chunk` — SSE payload for ``/v1/responses`` matching the
  existing wire format used by ``backend/server.py:953-1042``.
- Chat Completions helper chunks for the public ``/v1/chat/completions`` API.
- :func:`to_audit` — sanitized internal event for ``audit_events``.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from backend.agents_sdk.hitl import InterruptionEnvelope, interruption_from_sdk_item
from backend.audit.store import (
    CATEGORY_AGENT_HANDOFF,
    CATEGORY_HITL_PAUSE,
    CATEGORY_LLM_CHUNK,
    CATEGORY_RUN_COMPLETE,
    CATEGORY_TOOL_CALL,
    CATEGORY_TOOL_RESULT,
)


_PLACEHOLDER_FC_ID_RE = re.compile(
    r'^(?:__fake_id__|fake_id|placeholder|tmp|temp|undefined|null)$', re.IGNORECASE,
)
_FINAL_REPORT_TOOL_NAME = 'final_report'
_FINAL_REPORT_FIELD = 'report_markdown'
_FINAL_REPORT_METADATA_KEY = 'pai_final_report'
_FINAL_REPORT_FIELD_RE = re.compile(r'"report_markdown"\s*:\s*"')

# ``GenericHandler.do_*`` returns a ``StepOutcome`` whose ``next_prompt`` carries
# the working-memory anchor (``### [WORKING MEMORY]`` / ``<history>``) the legacy
# loop injects as a follow-up user message. The SDK has no separate follow-up
# channel, so ``backend/tools/wrappers.py:_outcome_to_str`` concatenates it onto
# the tool-result string. The model needs that scaffolding (it carries the
# evolving working memory), but business clients reading
# ``function_call_output.output`` should see only the actual tool result.
# Strip everything from the marker to end of string at the wire layer.
_INTERNAL_ANCHOR_RE = re.compile(
    r'\n*###\s*\[WORKING MEMORY\][\s\S]*\Z',
    re.IGNORECASE,
)


def _validate_arguments_json(arguments: str) -> tuple[bool, str | None]:
    """Check whether a ``function_call.arguments`` string parses as JSON.

    Returns ``(ok, error_message)``. Empty string and ``"{}"`` count as valid.
    Non-string input is treated as invalid because the wire contract requires
    ``arguments`` to be a serialized JSON string.

    qwen-plus and similar providers occasionally truncate streamed arguments
    mid-token; emitting that as ``status: 'completed'`` causes strict clients
    (OpenAI SDK ``parse()``) to throw on ``JSON.parse``. We mark such items
    ``status: 'incomplete'`` + ``arguments_status: 'invalid_json'`` so callers
    can detect and skip without crashing.
    """
    if not isinstance(arguments, str):
        return False, 'arguments must be a JSON string'
    if not arguments.strip():
        return True, None
    try:
        json.loads(arguments)
    except (TypeError, ValueError) as exc:
        return False, str(exc)
    return True, None


def _apply_arguments_validation(entry: dict, arguments: str) -> dict:
    """Attach ``arguments_status`` + downgrade ``status`` if JSON is malformed.

    Returns the same ``entry`` mutated in place. Safe to call repeatedly with
    the same arguments — the result is idempotent.
    """
    ok, _ = _validate_arguments_json(arguments)
    if ok:
        entry.pop('arguments_status', None)
    else:
        entry['status'] = 'incomplete'
        entry['arguments_status'] = 'invalid_json'
    return entry


def _strip_internal_markup(text: str) -> str:
    """Remove the working-memory anchor block from a tool-output string.

    Only operates on the wire-emitted ``function_call_output.output``. The
    model still receives the full string via the SDK's conversation history,
    so behavior on subsequent turns is unchanged.
    """
    if not text:
        return text
    cleaned = _INTERNAL_ANCHOR_RE.sub('', text)
    return cleaned.rstrip()


def _is_placeholder_fc_id(value: str) -> bool:
    return not value or bool(_PLACEHOLDER_FC_ID_RE.match(value))


# Internal protocol tags emitted by the model that must NOT leak to clients.
# Only ``<summary>`` is stripped at the wire layer — it is pure metadata used
# by the runner for history-summarisation retries and has no user-facing
# meaning. The reasoning-style tags (``<thinking>``, ``<checking>``,
# ``<taking>``, ``<working>``, ``<clinical-thinking>``, ``<taking-action>``,
# ``<skill-context>``) are intentionally passed through: the React frontend's
# ``consumeTextDelta`` parser converts them into ``thought_delta`` updates
# that render as the "Thinking…" panel. Stripping them here would leave that
# panel empty.
_HIDDEN_TAG_NAMES: tuple[str, ...] = ('summary',)
# Worst case lookahead: longest "<tagname" + 1 char of attr-or-close lookahead.
_HIDDEN_TAG_LOOKAHEAD = max(len(t) for t in _HIDDEN_TAG_NAMES) + 2


class StreamProtocolFilter:
    """Stateful filter that drops ``<summary>...</summary>`` and similar
    internal protocol blocks from streamed assistant text.

    Tag boundaries can split across SSE deltas (``"<sum"`` / ``"mary>"``), so
    the filter buffers ambiguous prefixes until it has enough lookahead to
    decide whether a ``<`` opens a hidden tag or is just literal text.

    Behavior:
    - When inside a hidden tag, all bytes are dropped until the matching
      ``</tag>`` is seen (case-insensitive).
    - When not inside a hidden tag, text up to the next ``<`` is forwarded
      immediately; the ``<`` itself is held until either it can be confirmed
      as a hidden-tag opener (then dropped through the closing tag) or as
      literal text (then forwarded).
    - ``flush()`` drains any remaining buffer at end-of-message. An unclosed
      hidden tag is dropped silently — better than leaking a half-tag.
    """

    __slots__ = ('_buffer', '_in_tag')

    def __init__(self) -> None:
        self._buffer: str = ''
        self._in_tag: str | None = None

    def feed(self, delta: str) -> str:
        if not delta:
            return ''
        self._buffer += delta
        out: list[str] = []
        while self._buffer:
            if self._in_tag is not None:
                close = f'</{self._in_tag}>'
                buf_lower = self._buffer.lower()
                close_lower = close.lower()
                idx = buf_lower.find(close_lower)
                if idx < 0:
                    # Closing tag not yet present. Retain the longest tail of
                    # the buffer that could be the start of ``</tag>`` so the
                    # split close-tag is recognized once the rest arrives.
                    keep = 0
                    max_keep = min(len(self._buffer), len(close) - 1)
                    for k in range(max_keep, 0, -1):
                        if buf_lower.endswith(close_lower[:k]):
                            keep = k
                            break
                    self._buffer = self._buffer[len(self._buffer) - keep:] if keep else ''
                    return ''.join(out)
                self._buffer = self._buffer[idx + len(close):]
                self._in_tag = None
                continue
            lt = self._buffer.find('<')
            if lt < 0:
                out.append(self._buffer)
                self._buffer = ''
                break
            if lt > 0:
                out.append(self._buffer[:lt])
                self._buffer = self._buffer[lt:]
            # Buffer now starts with '<'. Decide if it's a hidden tag.
            matched_tag: str | None = None
            for tag in _HIDDEN_TAG_NAMES:
                opener = f'<{tag}'
                if self._buffer.lower().startswith(opener.lower()):
                    nxt = self._buffer[len(opener):len(opener) + 1]
                    if nxt and nxt not in (' ', '>', '\t', '\n', '\r', '/'):
                        # Looks like ``<thinking-extra>`` — not our tag.
                        continue
                    matched_tag = tag
                    break
            if matched_tag is not None:
                close_open = self._buffer.find('>')
                if close_open < 0:
                    # Opening tag not yet complete — wait for more data.
                    return ''.join(out)
                self._in_tag = matched_tag
                self._buffer = self._buffer[close_open + 1:]
                continue
            # '<' is not a confirmed hidden-tag opener. If the buffer is too
            # short to rule one out, hold and wait for more bytes.
            if len(self._buffer) < _HIDDEN_TAG_LOOKAHEAD:
                possible = False
                lower = self._buffer.lower()
                for tag in _HIDDEN_TAG_NAMES:
                    opener = f'<{tag}'
                    if opener.lower().startswith(lower):
                        possible = True
                        break
                if possible:
                    return ''.join(out)
            out.append('<')
            self._buffer = self._buffer[1:]
        return ''.join(out)

    def flush(self) -> str:
        if self._in_tag is not None:
            self._buffer = ''
            self._in_tag = None
            return ''
        out = self._buffer
        self._buffer = ''
        # If the leftover is a strict prefix of any hidden-tag opener
        # (``<sum`` / ``<summary`` / ``<thinking``), we never received the
        # rest. Drop conservatively so a half-tag doesn't leak.
        lower = out.lower()
        for tag in _HIDDEN_TAG_NAMES:
            opener = f'<{tag}'.lower()
            if lower and len(lower) <= len(opener) and opener.startswith(lower):
                return ''
        return out


@dataclass
class ResponsesStreamState:
    """Tracks output index + accumulated text across a single response stream.

    The Responses SSE wire requires ``output_index`` to monotonically grow as
    output items are added, and a final ``response.completed`` carries the
    full ``output`` array.

    ``emitted_item_ids`` dedupes function_call items: when the upstream
    provider streams ``response.output_item.added(function_call)`` (and the
    accompanying ``function_call_arguments.delta/done``) on the raw wire, we
    forward those directly. The SDK *also* fires a ``RunItemStreamEvent``
    with a ``ToolCallItem`` for the same call; ``_item_to_responses`` uses
    this set to skip the synthesized added/done so clients don't see a
    duplicate card.

    ``current_step_id`` / ``step_counter`` track synthesized reasoning-step
    boundaries — the SDK has no native reasoning event, so we bracket each
    chunk of assistant thinking (i.e. text segments before a tool call) with
    ``response.reasoning_step.started`` / ``.completed`` carrying an
    ``rs_synth_*`` step id. Clients can use the ``rs_synth_`` prefix to
    distinguish our synthetic boundaries from a future native o1/o3
    ``reasoning_item`` if/when SDK begins forwarding those.

    ``current_fc_idx`` / ``current_fc_id`` track the function_call currently
    being built. Some upstreams (e.g. qwen-plus) reuse ``__fake_id__`` for
    every function_call's ``id`` field, so matching arguments.delta/done back
    to a state.output entry by id alone collides — every later tool's args
    overwrite the first tool. We rewrite the id to ``call_id`` on add, and
    use the index to attribute subsequent arguments events to the correct
    entry regardless of what placeholder the upstream sends.
    """
    output: list[dict] = field(default_factory=list)
    accumulated_text: list[str] = field(default_factory=list)
    message_started: bool = False
    message_index: int | None = None
    message_id: str | None = None
    emitted_item_ids: set[str] = field(default_factory=set)
    current_step_id: str | None = None
    step_counter: int = 0
    current_fc_idx: int | None = None
    current_fc_id: str | None = None
    final_report_message_started: bool = False
    final_report_message_index: int | None = None
    final_report_message_id: str | None = None
    final_report_args_text: str = ''
    final_report_streamed_text: str = ''
    protocol_filter: StreamProtocolFilter = field(default_factory=StreamProtocolFilter)


def _open_step_if_needed(state: ResponsesStreamState, *, response_id: str) -> list[dict]:
    if state.current_step_id is not None:
        return []
    state.step_counter += 1
    state.current_step_id = f'rs_synth_{response_id}_{state.step_counter}'
    return [{
        'type': 'response.reasoning_step.started',
        'response_id': response_id,
        'step_id': state.current_step_id,
        'synthetic': True,
    }]


def _close_step_if_open(state: ResponsesStreamState, *, response_id: str) -> list[dict]:
    if state.current_step_id is None:
        return []
    chunk = {
        'type': 'response.reasoning_step.completed',
        'response_id': response_id,
        'step_id': state.current_step_id,
        'synthetic': True,
    }
    state.current_step_id = None
    return [chunk]


def _current_function_name(state: ResponsesStreamState) -> str:
    idx = state.current_fc_idx
    if idx is None or idx < 0 or idx >= len(state.output):
        return ''
    entry = state.output[idx]
    if not isinstance(entry, dict) or entry.get('type') != 'function_call':
        return ''
    return str(entry.get('name') or '')


def _stream_text_chunks(text: str, *, max_chars: int = 96) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        split = end
        if end < len(text):
            window = text[start:end]
            for sep in ('\n\n', '\n', '。', '. ', '；', '; ', '，', ', '):
                pos = window.rfind(sep)
                if pos >= max(16, len(window) // 2):
                    split = start + pos + len(sep)
                    break
        chunks.append(text[start:split])
        start = split
    return chunks


def _ensure_final_report_message(state: ResponsesStreamState, *, response_id: str) -> list[dict]:
    if state.final_report_message_started and state.final_report_message_index is not None:
        return []
    state.final_report_message_started = True
    state.final_report_message_id = f'msg_final_{uuid.uuid4().hex}'
    state.final_report_message_index = len(state.output)
    state.output.append({
        'id': state.final_report_message_id,
        'type': 'message',
        'status': 'in_progress',
        'role': 'assistant',
        'content': [{'type': 'output_text', 'text': ''}],
        'metadata': {_FINAL_REPORT_METADATA_KEY: True},
    })
    return [{
        'type': 'response.output_item.added',
        'response_id': response_id,
        'output_index': state.final_report_message_index,
        'item': dict(state.output[state.final_report_message_index]),
    }]


def _stream_final_report_text(
    state: ResponsesStreamState,
    *,
    response_id: str,
    target_text: str,
) -> list[dict]:
    if not target_text.startswith(state.final_report_streamed_text):
        return []
    delta = target_text[len(state.final_report_streamed_text):]
    if not delta:
        return []
    chunks = _ensure_final_report_message(state, response_id=response_id)
    idx = state.final_report_message_index
    if idx is None:
        return chunks
    for piece in _stream_text_chunks(delta):
        if not piece:
            continue
        state.final_report_streamed_text += piece
        state.output[idx]['content'] = [{'type': 'output_text', 'text': state.final_report_streamed_text}]
        chunks.append({
            'type': 'response.output_text.delta',
            'response_id': response_id,
            'delta': piece,
            'output_index': idx,
            'content_index': 0,
        })
    return chunks


def _complete_final_report_message(
    state: ResponsesStreamState,
    *,
    response_id: str,
    final_text: str,
) -> list[dict]:
    chunks = _stream_final_report_text(state, response_id=response_id, target_text=final_text)
    if not state.final_report_message_started or state.final_report_message_index is None:
        return chunks
    idx = state.final_report_message_index
    state.final_report_streamed_text = final_text
    completed = {
        'id': state.final_report_message_id,
        'type': 'message',
        'status': 'completed',
        'role': 'assistant',
        'content': [{'type': 'output_text', 'text': final_text}],
        'metadata': {_FINAL_REPORT_METADATA_KEY: True},
    }
    state.output[idx] = completed
    chunks.extend([
        {
            'type': 'response.output_text.done',
            'response_id': response_id,
            'text': final_text,
            'output_index': idx,
            'content_index': 0,
        },
        {
            'type': 'response.output_item.done',
            'response_id': response_id,
            'output_index': idx,
            'item': dict(completed),
        },
    ])
    return chunks


def _extract_report_markdown_prefix(arguments_text: str) -> str:
    match = _FINAL_REPORT_FIELD_RE.search(arguments_text or '')
    if not match:
        return ''
    raw = arguments_text[match.end():]
    out: list[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == '"':
            break
        if ch != '\\':
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= len(raw):
            break
        esc = raw[i]
        if esc == 'n':
            out.append('\n')
        elif esc == 'r':
            out.append('\r')
        elif esc == 't':
            out.append('\t')
        elif esc == 'b':
            out.append('\b')
        elif esc == 'f':
            out.append('\f')
        elif esc == 'u':
            hex_digits = raw[i + 1:i + 5]
            if len(hex_digits) < 4 or not re.fullmatch(r'[0-9a-fA-F]{4}', hex_digits):
                break
            out.append(chr(int(hex_digits, 16)))
            i += 4
        else:
            out.append(esc)
        i += 1
    return ''.join(out)


def _model_dump(obj: Any) -> dict:
    if hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump(exclude_unset=True)
        except Exception:
            pass
    if isinstance(obj, dict):
        return dict(obj)
    return {}


def _event_kind(event: Any) -> str:
    """Return SDK event discriminator: 'raw', 'item', 'agent', or 'unknown'."""
    t = getattr(event, 'type', None)
    if t == 'raw_response_event':
        return 'raw'
    if t == 'run_item_stream_event':
        return 'item'
    if t == 'agent_updated_stream_event':
        return 'agent'
    return 'unknown'


def _item_kind(item: Any) -> str:
    return type(item).__name__


def to_responses_chunk(sdk_event: Any, state: ResponsesStreamState, *, response_id: str) -> list[dict]:
    """Translate one SDK ``StreamEvent`` to zero-or-more Responses-wire chunks.

    Returns a list because some SDK events expand into multiple wire chunks
    (e.g. ``ToolCallItem`` → ``output_item.added`` + ``output_item.done``).
    """
    kind = _event_kind(sdk_event)
    if kind == 'raw':
        return _raw_to_responses(sdk_event.data, state, response_id=response_id)
    if kind == 'item':
        return _item_to_responses(sdk_event.item, sdk_event.name, state, response_id=response_id)
    return []


def _raw_to_responses(data: Any, state: ResponsesStreamState, *, response_id: str) -> list[dict]:
    """Handle ``RawResponsesStreamEvent.data`` — these are the upstream OpenAI
    Responses-API stream events. Three categories survive translation:

    - ``response.output_text.delta`` opens (synthesizes) the assistant
      message card and streams text. First delta of a turn also opens a
      synthetic reasoning step.
    - ``response.function_call_arguments.delta/done`` and the matching raw
      ``output_item.added/done(function_call)`` are forwarded so clients
      can render argument streaming. We register the item id in
      ``emitted_item_ids`` so the eventual ``ToolCallItem`` doesn't double
      up. A function_call appearing here also closes any open reasoning
      step (the model handed off to a tool).
    - Everything else is dropped — we don't proxy upstream's full event
      vocabulary, only what clients actually consume.
    """
    type_name = getattr(data, 'type', '') or ''

    if type_name == 'response.output_text.delta':
        delta = getattr(data, 'delta', '') or ''
        # Skip noise: empty/whitespace deltas would otherwise create an empty
        # leading ``message`` item in ``response.output``. Strict OpenAI clients
        # treat that as an empty assistant turn.
        if not delta:
            return []
        # Strip internal protocol blocks (``<summary>...</summary>`` etc.) at
        # the wire layer so streaming clients never see meta-tags. The filter
        # is stateful — partial tags split across deltas are buffered until
        # they can be classified.
        visible_delta = state.protocol_filter.feed(delta)
        if not visible_delta:
            return []
        chunks: list[dict] = []
        chunks.extend(_open_step_if_needed(state, response_id=response_id))
        if not state.message_started:
            state.message_started = True
            state.message_id = f'msg_{uuid.uuid4().hex}'
            state.message_index = len(state.output)
            state.output.append({
                'id': state.message_id,
                'type': 'message',
                'status': 'in_progress',
                'role': 'assistant',
                'content': [],
            })
            chunks.append({
                'type': 'response.output_item.added',
                'response_id': response_id,
                'output_index': state.message_index,
                'item': dict(state.output[state.message_index]),
            })
        state.accumulated_text.append(visible_delta)
        chunks.append({
            'type': 'response.output_text.delta',
            'response_id': response_id,
            'delta': visible_delta,
            'output_index': state.message_index,
            'content_index': 0,
        })
        return chunks

    if type_name == 'response.output_item.added':
        item = _model_dump(getattr(data, 'item', None))
        if item.get('type') == 'function_call':
            chunks = []
            chunks.extend(_close_step_if_open(state, response_id=response_id))
            raw_id = str(item.get('id') or '')
            call_id = str(item.get('call_id') or '')
            # Rewrite placeholder ids so each function_call is uniquely
            # addressable. Without this, qwen-plus and similar providers
            # send `id="__fake_id__"` for every call, and downstream
            # by-id matching collides across tools.
            if _is_placeholder_fc_id(raw_id):
                rewritten = call_id or f'fc_{uuid.uuid4().hex}'
                item['id'] = rewritten
            new_id = str(item.get('id') or '')
            if new_id:
                state.emitted_item_ids.add(new_id)
            if call_id:
                state.emitted_item_ids.add(call_id)
            idx = len(state.output)
            state.output.append(dict(item))
            state.current_fc_idx = idx
            state.current_fc_id = new_id
            if item.get('name') == _FINAL_REPORT_TOOL_NAME:
                state.final_report_args_text = str(item.get('arguments') or '')
                state.final_report_streamed_text = ''
                state.final_report_message_started = False
                state.final_report_message_index = None
                state.final_report_message_id = None
            chunks.append({
                'type': 'response.output_item.added',
                'response_id': response_id,
                'output_index': idx,
                'item': dict(item),
            })
            return chunks
        return []

    if type_name == 'response.function_call_arguments.delta':
        # Substitute the placeholder item_id with our rewritten id so the
        # client can correlate arguments.delta to the correct tool call.
        upstream_id = getattr(data, 'item_id', '') or ''
        item_id = state.current_fc_id or upstream_id
        chunks = [{
            'type': 'response.function_call_arguments.delta',
            'response_id': response_id,
            'item_id': item_id,
            'output_index': getattr(data, 'output_index', 0) or 0,
            'delta': getattr(data, 'delta', '') or '',
        }]
        if _current_function_name(state) == _FINAL_REPORT_TOOL_NAME:
            delta = getattr(data, 'delta', '') or ''
            state.final_report_args_text += delta
            target = _extract_report_markdown_prefix(state.final_report_args_text)
            chunks.extend(_stream_final_report_text(state, response_id=response_id, target_text=target))
        return chunks

    if type_name == 'response.function_call_arguments.done':
        # Final arguments string for an item streamed via raw deltas. Update
        # the matching state.output entry so the terminal ``response.completed``
        # carries the full arguments, not the empty placeholder we inserted on
        # ``output_item.added``. Match by current_fc_idx (set on add) since
        # upstream item_id can be a duplicated placeholder.
        full_args = getattr(data, 'arguments', '') or ''
        args_ok, _ = _validate_arguments_json(full_args)
        idx = state.current_fc_idx
        if idx is not None and 0 <= idx < len(state.output):
            entry = state.output[idx]
            if entry.get('type') == 'function_call':
                entry['arguments'] = full_args
                entry['status'] = 'completed'
                _apply_arguments_validation(entry, full_args)
        item_id = state.current_fc_id or (getattr(data, 'item_id', '') or '')
        done_chunk: dict = {
            'type': 'response.function_call_arguments.done',
            'response_id': response_id,
            'item_id': item_id,
            'output_index': getattr(data, 'output_index', 0) or 0,
            'arguments': full_args,
        }
        if not args_ok:
            done_chunk['arguments_status'] = 'invalid_json'
        chunks = [done_chunk]
        if _current_function_name(state) == _FINAL_REPORT_TOOL_NAME:
            state.final_report_args_text = full_args
            final_report = _extract_report_markdown_prefix(full_args)
            chunks.extend(_complete_final_report_message(
                state,
                response_id=response_id,
                final_text=final_report,
            ))
        return chunks

    if type_name == 'response.output_item.done':
        item = _model_dump(getattr(data, 'item', None))
        if item.get('type') == 'function_call':
            raw_id = str(item.get('id') or '')
            if _is_placeholder_fc_id(raw_id) and state.current_fc_id:
                item['id'] = state.current_fc_id
            idx = state.current_fc_idx
            if idx is not None and 0 <= idx < len(state.output):
                entry = state.output[idx]
                if entry.get('type') == 'function_call':
                    entry['arguments'] = item.get('arguments', entry.get('arguments', ''))
                    entry['status'] = 'completed'
                    _apply_arguments_validation(entry, entry.get('arguments', '') or '')
                    item['status'] = entry['status']
                    if 'arguments_status' in entry:
                        item['arguments_status'] = entry['arguments_status']
                    else:
                        item.pop('arguments_status', None)
                output_index = idx
            else:
                output_index = 0
                for i, entry in enumerate(state.output):
                    if entry.get('id') == item.get('id') and entry.get('type') == 'function_call':
                        output_index = i
                        break
            chunks = [{
                'type': 'response.output_item.done',
                'response_id': response_id,
                'output_index': output_index,
                'item': dict(item),
            }]
            state.current_fc_idx = None
            state.current_fc_id = None
            return chunks
        return []

    return []


def _item_to_responses(item: Any, name: str, state: ResponsesStreamState, *, response_id: str) -> list[dict]:
    kind = _item_kind(item)
    if kind == 'ToolCallItem':
        call = _tool_call_payload(item)
        # Dedup: if raw stream already forwarded output_item.added for this
        # call (raw streaming path), suppress synthesized added/done. The raw
        # path's ``output_item.done`` (or function_call_arguments.done) has
        # already finalized arguments + state.output entry.
        # Check both id and call_id — qwen-plus-style providers reuse
        # ``__fake_id__`` as raw_item.id, so id alone misses the dedup; the
        # raw path also tracks call_id for exactly this case.
        raw_id = getattr(item, 'raw_item', None)
        candidate_ids: list[str] = []
        if isinstance(raw_id, dict):
            for key in ('id', 'call_id'):
                v = raw_id.get(key)
                if v:
                    candidate_ids.append(str(v))
        elif raw_id is not None:
            for key in ('id', 'call_id'):
                v = getattr(raw_id, key, None)
                if v:
                    candidate_ids.append(str(v))
        # Also include the normalized call_id we extracted into the payload.
        if call.get('call_id'):
            candidate_ids.append(str(call['call_id']))
        if any(cid in state.emitted_item_ids for cid in candidate_ids):
            return []
        chunks = []
        # ToolCallItem appears at the end of an LLM "thinking" segment — close
        # any open reasoning step before emitting the tool card.
        chunks.extend(_close_step_if_open(state, response_id=response_id))
        idx = len(state.output)
        state.output.append(call)
        if call.get('id'):
            state.emitted_item_ids.add(str(call['id']))
        chunks.extend([
            {
                'type': 'response.output_item.added',
                'response_id': response_id,
                'output_index': idx,
                'item': dict(call),
            },
            {
                'type': 'response.output_item.done',
                'response_id': response_id,
                'output_index': idx,
                'item': dict(call),
            },
        ])
        return chunks
    if kind == 'ToolCallOutputItem':
        out = _tool_output_payload(item)
        idx = len(state.output)
        state.output.append(out)
        return [
            {
                'type': 'response.output_item.added',
                'response_id': response_id,
                'output_index': idx,
                'item': dict(out),
            },
            {
                'type': 'response.output_item.done',
                'response_id': response_id,
                'output_index': idx,
                'item': dict(out),
            },
        ]
    if kind == 'MessageOutputItem':
        # The raw text deltas already streamed; this is the closing "done"
        # signal for the assistant message. Mark it completed and close any
        # open reasoning step (text is the visible answer, not internal
        # thinking — the step ends here).
        # Drain any bytes the protocol filter is still holding (e.g. a
        # trailing literal '<' that hadn't been disambiguated against an
        # incomplete hidden-tag prefix) before finalizing the message.
        tail = state.protocol_filter.flush()
        tail_chunks: list[dict] = []
        if tail and state.message_started and state.message_index is not None:
            state.accumulated_text.append(tail)
            tail_chunks.append({
                'type': 'response.output_text.delta',
                'response_id': response_id,
                'delta': tail,
                'output_index': state.message_index,
                'content_index': 0,
            })
        if state.message_started and state.message_index is not None:
            done_text = ''.join(state.accumulated_text)
            # Defensive: if the run only emitted whitespace, remove the message
            # so ``response.output`` doesn't carry an empty assistant turn.
            # Pair with the input-side guard above that drops empty deltas.
            if not done_text.strip():
                state.output.pop(state.message_index)
                chunks = list(_close_step_if_open(state, response_id=response_id))
                state.message_started = False
                state.message_index = None
                state.message_id = None
                state.accumulated_text = []
                return chunks
            state.output[state.message_index] = {
                'id': state.message_id,
                'type': 'message',
                'status': 'completed',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': done_text}],
            }
            chunks = list(tail_chunks)
            chunks.extend([
                {
                    'type': 'response.output_text.done',
                    'response_id': response_id,
                    'text': done_text,
                    'output_index': state.message_index,
                    'content_index': 0,
                },
                {
                    'type': 'response.output_item.done',
                    'response_id': response_id,
                    'output_index': state.message_index,
                    'item': dict(state.output[state.message_index]),
                },
            ])
            chunks.extend(_close_step_if_open(state, response_id=response_id))
            # Reset for any subsequent assistant message in the same response.
            state.message_started = False
            state.message_index = None
            state.message_id = None
            state.accumulated_text = []
            return chunks
    return []


def _tool_call_payload(item: Any) -> dict:
    raw = getattr(item, 'raw_item', None)
    if isinstance(raw, dict):
        raw_dict = raw
    elif raw is not None:
        raw_dict = _model_dump(raw)
    else:
        raw_dict = {}
    call_id = (
        getattr(item, 'call_id', None)
        or raw_dict.get('call_id')
        or raw_dict.get('id')
        or f'call_{uuid.uuid4().hex}'
    )
    tool_name = getattr(item, 'tool_name', None) or raw_dict.get('name') or 'tool'
    args_raw = raw_dict.get('arguments') or '{}'
    if not isinstance(args_raw, str):
        args_raw = json.dumps(args_raw, ensure_ascii=False, default=str)
    raw_output_item_id = raw_dict.get('id') or ''
    if _is_placeholder_fc_id(str(raw_output_item_id)):
        # Never let `__fake_id__` (or similar) leak into state.output — derive
        # a stable id from call_id so downstream payloads stay clean.
        output_item_id = f'fc_{call_id}' if call_id else f'fc_{uuid.uuid4().hex}'
    else:
        output_item_id = raw_output_item_id
    payload = {
        'id': str(output_item_id),
        'type': 'function_call',
        'status': 'completed',
        'call_id': str(call_id),
        'name': str(tool_name),
        'arguments': args_raw,
    }
    return _apply_arguments_validation(payload, args_raw)


def _tool_output_payload(item: Any) -> dict:
    raw_item = getattr(item, 'raw_item', None)
    if isinstance(raw_item, dict):
        raw = raw_item
    elif raw_item is not None:
        raw = _model_dump(raw_item)
    else:
        raw = {}
    call_id = getattr(item, 'call_id', None) or raw.get('call_id') or ''
    output = raw.get('output')
    if output is None:
        output = getattr(item, 'output', '') or ''
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False, default=str)
    return {
        'id': f'fco_{uuid.uuid4().hex}',
        'type': 'function_call_output',
        'status': 'completed',
        'call_id': str(call_id),
        'output': _strip_internal_markup(output),
    }


def to_chat_chunk(sdk_event: Any, *, model: str, completion_id: str) -> dict | None:
    """Translate one SDK ``StreamEvent`` to a Chat Completions stream chunk.

    Only assistant text deltas are surfaced on the Chat wire. Server-side tool
    calls stay internal; HITL pauses are surfaced separately with
    ``chat_pause_chunk``.
    """
    import time as _t
    if _event_kind(sdk_event) != 'raw':
        return None
    data = sdk_event.data
    if getattr(data, 'type', '') != 'response.output_text.delta':
        return None
    delta = getattr(data, 'delta', '') or ''
    if not delta:
        return None
    return {
        'id': completion_id,
        'object': 'chat.completion.chunk',
        'created': int(_t.time()),
        'model': model,
        'choices': [{'index': 0, 'delta': {'content': delta}, 'finish_reason': None}],
    }


def chat_role_chunk(*, model: str, completion_id: str) -> dict:
    import time as _t
    return {
        'id': completion_id,
        'object': 'chat.completion.chunk',
        'created': int(_t.time()),
        'model': model,
        'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}],
    }


def chat_done_chunk(*, model: str, completion_id: str, usage: dict | None = None) -> dict:
    import time as _t
    chunk = {
        'id': completion_id,
        'object': 'chat.completion.chunk',
        'created': int(_t.time()),
        'model': model,
        'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
    }
    if usage:
        chunk['usage'] = dict(usage)
    return chunk


def chat_pause_chunk(envelope: 'InterruptionEnvelope', *, model: str, completion_id: str) -> dict:
    """Final assistant chunk that signals a HITL pause on the Chat wire."""
    import time as _t
    return {
        'id': completion_id,
        'object': 'chat.completion.chunk',
        'created': int(_t.time()),
        'model': model,
        'choices': [{
            'index': 0,
            'delta': {'tool_calls': [envelope.serialize('chat')]},
            'finish_reason': 'tool_calls',
        }],
    }


def to_audit(sdk_event: Any) -> dict | None:
    """Translate one SDK event to an audit row payload.

    Returns ``{'category': str, 'payload': dict}`` or ``None`` if the event
    isn't audit-worthy (handoffs, raw deltas we already record as chunks).
    """
    kind = _event_kind(sdk_event)
    if kind == 'raw':
        data = sdk_event.data
        if getattr(data, 'type', '') == 'response.output_text.delta':
            return {
                'category': CATEGORY_LLM_CHUNK,
                'payload': {'delta': getattr(data, 'delta', '') or ''},
            }
        return None
    if kind == 'item':
        item = sdk_event.item
        ikind = _item_kind(item)
        if ikind == 'ToolCallItem':
            return {
                'category': CATEGORY_TOOL_CALL,
                'payload': _tool_call_payload(item),
            }
        if ikind == 'ToolCallOutputItem':
            return {
                'category': CATEGORY_TOOL_RESULT,
                'payload': _tool_output_payload(item),
            }
        if ikind == 'ToolApprovalItem':
            envelope = interruption_from_sdk_item(item)
            return {
                'category': CATEGORY_HITL_PAUSE,
                'payload': {
                    'call_id': envelope.call_id,
                    'tool_name': envelope.tool_name,
                    'arguments': envelope.arguments,
                    'is_input_request': envelope.is_input_request,
                },
            }
    if kind == 'agent':
        return {
            'category': CATEGORY_AGENT_HANDOFF,
            'payload': {'new_agent': getattr(sdk_event.new_agent, 'name', '')},
        }
    return None


def interruption_envelope(sdk_event: Any) -> InterruptionEnvelope | None:
    """If this event surfaces a ToolApprovalItem, return its envelope."""
    if _event_kind(sdk_event) != 'item':
        return None
    item = sdk_event.item
    if _item_kind(item) != 'ToolApprovalItem':
        return None
    return interruption_from_sdk_item(item)


def run_complete_audit(*, final_text: str, usage: dict | None) -> dict:
    return {
        'category': CATEGORY_RUN_COMPLETE,
        'payload': {
            'final_text_chars': len(final_text or ''),
            'usage': usage or {},
        },
    }
