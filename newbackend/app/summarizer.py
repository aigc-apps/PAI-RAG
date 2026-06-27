from __future__ import annotations
from typing import Awaitable, Callable, List
from loguru import logger
from app.store.base import Item

CompleteFn = Callable[[str], Awaitable[str]]

_SUMMARY_INSTRUCTIONS = """You maintain a running summary of a conversation so older
turns can be dropped from context without losing important information.
Given the prior summary and the next batch of older messages, produce an UPDATED,
concise summary (a few short paragraphs max) that preserves durable facts, decisions,
open questions, and user intent. Do not include pleasantries or verbatim transcripts.
Output ONLY the updated summary text."""


def _item_line(it: Item) -> str:
    c = it.content or {}
    if it.type == "message":
        return f"{it.role or 'user'}: {c.get('text', '')}"
    if it.type == "function_call":
        return f"assistant called {c.get('name', '')}({c.get('arguments', '')})"
    if it.type == "function_call_output":
        return f"tool result: {c.get('output', '')}"
    return ""


class ConversationSummarizer:
    def __init__(self, complete: CompleteFn):
        self._complete = complete

    async def summarize(self, prior_summary: str, items: List[Item]) -> str:
        lines = "\n".join(line for line in (_item_line(it) for it in items) if line) or "(none)"
        prompt = (
            _SUMMARY_INSTRUCTIONS
            + "\n\nPrior summary:\n" + (prior_summary or "(none)")
            + "\n\nNew messages to fold in:\n" + lines
            + "\n\nUpdated summary:"
        )
        try:
            out = await self._complete(prompt)
        except Exception:
            logger.exception("conversation summarize failed")
            return prior_summary
        out = (out or "").strip()
        return out or prior_summary


async def maybe_summarize_conversation(
    store, conversation_id: str, complete: CompleteFn,
    keep_recent: int = 20, batch: int = 20,
) -> bool:
    """Fold old unsummarized items into the rolling summary when they overflow the
    kept-recent window. Returns True if a summary was written. Fully guarded."""
    try:
        conv = await store.get_conversation(conversation_id)
        if conv is None:
            return False
        items = await store.get_conversation_items(conversation_id)
        unsummarized = [it for it in items if it.seq > conv.summarized_seq]
        if len(unsummarized) <= keep_recent + batch:
            return False
        to_fold = unsummarized[:-keep_recent]  # all but the most recent keep_recent
        if not to_fold:
            return False
        fold_to_seq = to_fold[-1].seq
        new_summary = await ConversationSummarizer(complete).summarize(
            conv.summary or "", to_fold)
        if new_summary == (conv.summary or ""):
            # summarizer failed or produced nothing new — do NOT advance the cursor,
            # so the unsummarized items stay in history and are retried next turn.
            return False
        await store.update_conversation_summary(conversation_id, new_summary, fold_to_seq)
        return True
    except Exception:
        logger.exception("maybe_summarize_conversation failed")
        return False
