from __future__ import annotations
import json
import re
from typing import Awaitable, Callable, List, Optional
from loguru import logger
from app.store.base import MemoryItem

CompleteFn = Callable[[str], Awaitable[str]]

_EXTRACT_PROMPT = """You maintain a long-term memory about a specific user.
Given the latest exchange and the user's existing memories, decide what to change.
Return ONLY a JSON array of operations, each one of:
  {{"op":"ADD","text":"<new durable fact about the user>"}}
  {{"op":"UPDATE","target_id":"<id>","text":"<revised fact>"}}
  {{"op":"DELETE","target_id":"<id>"}}
  {{"op":"NOOP"}}
Only record durable, user-specific facts (preferences, identity, context). Do NOT
record transient chit-chat, the assistant's words, or anything sensitive the user
didn't volunteer. If nothing is worth changing, return [].

Existing memories:
{existing}

Latest exchange:
User: {user_text}
Assistant: {assistant_text}

JSON operations:"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    return m.group(1).strip() if m else s


class MemoryExtractor:
    def __init__(self, complete: CompleteFn):
        self._complete = complete

    async def extract(self, user_text: str, assistant_text: str,
                      existing: List[MemoryItem]) -> List[dict]:
        existing_str = "\n".join(f'- (id={m.id}) {m.text}' for m in existing) or "(none)"
        prompt = _EXTRACT_PROMPT.format(
            existing=existing_str, user_text=user_text, assistant_text=assistant_text)
        try:
            raw = await self._complete(prompt)
        except Exception:
            logger.exception("memory extract: completion failed")
            return []
        try:
            parsed = json.loads(_strip_fences(raw))
        except Exception:
            logger.warning(f"memory extract: non-JSON output: {raw[:200]!r}")
            return []
        if not isinstance(parsed, list):
            return []
        return [op for op in parsed if isinstance(op, dict) and "op" in op]


async def apply_memory_ops(store, user_id: str, ops: List[dict],
                           source_response_id: Optional[str] = None) -> None:
    for op in ops:
        kind = op.get("op")
        try:
            if kind == "ADD" and op.get("text"):
                await store.add_memory(MemoryItem(
                    user_id=user_id, text=op["text"], source_response_id=source_response_id))
            elif kind == "UPDATE" and op.get("target_id") and op.get("text"):
                await store.update_memory(op["target_id"], op["text"])
            elif kind == "DELETE" and op.get("target_id"):
                await store.delete_memory(op["target_id"])
            # NOOP / unknown: skip
        except Exception:
            logger.exception(f"memory apply: op failed: {op}")


async def update_user_memory(store, user_id: str, user_text: str, assistant_text: str,
                             complete: CompleteFn,
                             source_response_id: Optional[str] = None) -> None:
    """List existing -> extract -> apply. Fully guarded; never raises."""
    try:
        existing = await store.list_memories(user_id)
        ops = await MemoryExtractor(complete).extract(user_text, assistant_text, existing)
        if ops:
            await apply_memory_ops(store, user_id, ops, source_response_id)
    except Exception:
        logger.exception("update_user_memory failed")


def make_complete(llm) -> CompleteFn:
    """Single-shot completion over a LeanLLM-style astream (collects deltas)."""
    async def complete(prompt: str) -> str:
        chunks: List[str] = []
        async for ch in llm.astream(messages=[{"role": "user", "content": prompt}], tools=[]):
            delta = getattr(ch, "delta", "") or ""
            if delta:
                chunks.append(delta)
        return "".join(chunks)
    return complete
