from __future__ import annotations
from typing import Dict, List, Optional
from app.store.base import Item, StoredResponse


def _text(content: dict) -> str:
    return (content or {}).get("text", "") or ""


def _timeline(content: dict) -> Optional[List[Dict[str, str]]]:
    value = (content or {}).get("timeline")
    if not isinstance(value, list) or not value:
        return None
    steps: List[Dict[str, str]] = []
    for step in value:
        if not isinstance(step, dict):
            return None
        kind = step.get("kind")
        if kind in ("reasoning", "text"):
            text = step.get("text")
            if not isinstance(text, str):
                return None
            steps.append({"kind": kind, "text": text})
        elif kind == "tool":
            call_id = step.get("id")
            if not isinstance(call_id, str) or not call_id:
                return None
            steps.append({"kind": "tool", "id": call_id})
        else:
            return None
    return steps


def group_conversation_messages(
    items: List[Item], responses: List[StoredResponse]
) -> List[Dict]:
    """Reconstruct UI messages from the per-turn item log.

    Items are appended per turn; every item of a turn shares one ``response_id``
    (user ``message`` + optional ``reasoning`` + ``function_call``/output +
    assistant ``message``). We group by ``response_id`` preserving first-seen
    ``seq`` order, then yield up to two UI messages per group: a ``user`` message
    and an ``assistant`` message carrying reasoning + the turn's response status.
    ``function_call``/``function_call_output`` items are skipped (tool UI is out
    of scope for v1).
    """
    by_id: Dict[str, StoredResponse] = {r.id: r for r in responses}

    # Group items by response_id, preserving chronological (seq) first-seen order.
    order: List[str] = []
    groups: Dict[str, List[Item]] = {}
    for it in sorted(items, key=lambda i: i.seq):
        key = it.response_id or ""
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(it)

    messages: List[Dict] = []
    for key in order:
        group = groups[key]
        user_item = next(
            (i for i in group if i.type == "message" and i.role == "user"), None
        )
        assistant_item = next(
            (i for i in group if i.type == "message" and i.role == "assistant"), None
        )
        reasoning_item = next((i for i in group if i.type == "reasoning"), None)
        resp = by_id.get(key)

        if user_item is not None:
            messages.append(
                {"role": "user", "text": _text(user_item.content), "response_id": key}
            )

        # Emit an assistant message whenever the group corresponds to a response
        # turn (has a response row, or any assistant/reasoning content). A failed
        # turn has a response row but no assistant message item -> empty text.
        if resp is not None or assistant_item is not None or reasoning_item is not None:
            fcalls = [i for i in group if i.type == "function_call"]
            outputs = {i.content.get("call_id"): i.content
                       for i in group if i.type == "function_call_output"}
            tool_calls = [
                {
                    "call_id": c.content.get("call_id", ""),
                    "name": c.content.get("name", ""),
                    "arguments": c.content.get("arguments", "") or "",
                    "output": (outputs.get(c.content.get("call_id")) or {}).get("output", ""),
                    "files": (outputs.get(c.content.get("call_id")) or {}).get("files", []),
                    # Persisted HITL notice (e.g. aliyun authorization card); None
                    # for ordinary tools and pre-change history.
                    "notice": (outputs.get(c.content.get("call_id")) or {}).get("notice"),
                }
                for c in fcalls
            ]
            assistant_message = {
                "role": "assistant",
                "text": _text(assistant_item.content) if assistant_item else "",
                "reasoning": _text(reasoning_item.content) if reasoning_item else None,
                "response_id": key,
                "previous_response_id": resp.previous_response_id if resp else None,
                "status": resp.status if resp else "completed",
                "tool_calls": tool_calls,
                "usage": resp.usage if resp else None,
            }
            timeline = _timeline(assistant_item.content) if assistant_item else None
            if timeline is not None:
                assistant_message["steps"] = timeline
            messages.append(assistant_message)
    return messages
