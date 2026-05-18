"""Map SDK interruptions ↔ OpenAI wire formats.

Two wires:

- **responses** (``/v1/responses``): native ``requires_action`` with a
  ``function_call`` output item the client resolves by posting back a
  ``function_call_output`` input item.
- **chat** (``/v1/chat/completions``): synthetic ``tool_calls`` with a
  reserved name; client resolves by appending a ``role:"tool"`` message.

Reserved tool names on the chat wire:

- ``__ask_user__`` — pause to ask the user a clarifying question.
- ``__request_approval__`` — pause to get approval before calling a sensitive
  tool (file_write, code_run, …) gated by ``needs_approval=True``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

WIRE_RESPONSES = 'responses'
WIRE_CHAT = 'chat'

RESERVED_ASK_USER = '__ask_user__'
RESERVED_REQUEST_APPROVAL = '__request_approval__'


@dataclass
class InterruptionEnvelope:
    """Wire-agnostic description of one paused tool call.

    ``call_id`` is the SDK's interruption item id; clients echo it back on
    resume so we can match the answer to the right interruption.
    """
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    is_input_request: bool  # True for ask_user, False for generic approvals

    def serialize(self, wire: str) -> dict[str, Any]:
        if wire == WIRE_RESPONSES:
            return {
                'type': 'function_call',
                'status': 'in_progress',
                'call_id': self.call_id,
                'name': self.tool_name,
                'arguments': json.dumps(self.arguments, ensure_ascii=False),
            }
        if wire == WIRE_CHAT:
            display_name = RESERVED_ASK_USER if self.is_input_request else RESERVED_REQUEST_APPROVAL
            return {
                'id': self.call_id,
                'type': 'function',
                'function': {
                    'name': display_name,
                    'arguments': json.dumps(
                        {'tool': self.tool_name, **self.arguments}, ensure_ascii=False,
                    ),
                },
            }
        raise ValueError(f'unknown wire: {wire!r}')


@dataclass
class ResumePayload:
    """Parsed resume input from either wire."""
    call_id: str
    answer: str
    approve: bool  # always True for ask_user; can be False for explicit reject

    @classmethod
    def from_responses_input(cls, input_items: list[dict[str, Any]]) -> 'ResumePayload':
        for item in input_items:
            if item.get('type') == 'function_call_output' and item.get('call_id'):
                return cls(
                    call_id=item['call_id'],
                    answer=str(item.get('output', '')),
                    approve=True,
                )
            if item.get('type') == 'mcp_approval_response' and item.get('approval_request_id'):
                return cls(
                    call_id=item['approval_request_id'],
                    answer=str(item.get('output', '')),
                    approve=bool(item.get('approve', True)),
                )
        raise ValueError('no function_call_output / mcp_approval_response item found')

    @classmethod
    def from_chat_message(cls, message: dict[str, Any]) -> 'ResumePayload':
        if message.get('role') != 'tool':
            raise ValueError("expected role='tool' resume message")
        call_id = message.get('tool_call_id')
        if not call_id:
            raise ValueError('tool message missing tool_call_id')
        content = message.get('content') or ''
        if isinstance(content, list):
            content = ''.join(
                p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text'
            )
        return cls(call_id=str(call_id), answer=str(content), approve=True)


def interruption_from_sdk_item(item: Any) -> InterruptionEnvelope:
    """Adapt a SDK ``ToolApprovalItem`` to the wire-agnostic envelope.

    Kept narrow: only reads the public attributes (``raw_item``/``tool_name``/
    ``arguments``) so this still works if the SDK extends the item type.
    """
    raw_item = getattr(item, 'raw_item', None)
    if isinstance(raw_item, dict):
        raw = raw_item
    elif raw_item is not None and hasattr(raw_item, 'model_dump'):
        try:
            raw = raw_item.model_dump(exclude_unset=True)
        except Exception:
            raw = {}
    else:
        raw = {}
    call_id = getattr(item, 'id', None) or raw.get('call_id') or raw.get('id') or ''
    tool_name = getattr(item, 'tool_name', None) or raw.get('name') or ''
    args_raw = getattr(item, 'arguments', None)
    if args_raw is None:
        args_raw = raw.get('arguments') or '{}'
    if isinstance(args_raw, str):
        try:
            arguments = json.loads(args_raw)
        except json.JSONDecodeError:
            arguments = {'_raw': args_raw}
    elif isinstance(args_raw, dict):
        arguments = args_raw
    else:
        arguments = {}
    is_input_request = tool_name == 'ask_user'
    return InterruptionEnvelope(
        call_id=str(call_id),
        tool_name=str(tool_name),
        arguments=arguments,
        is_input_request=is_input_request,
    )
