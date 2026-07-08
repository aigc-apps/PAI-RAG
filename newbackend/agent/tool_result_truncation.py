"""Rule-based structural truncation for oversized tool results.

The copy of a tool result that is fed to the LLM (``agent/agent.py`` →
``AgentMessageManager.cap_tool_result``) used to be head-only truncated: a
100k-token result became its first ~5k tokens and the *tail* — shell exit
codes, a list's ``TotalCount``, the last traceback frame, the final record —
was dropped, so the model never saw "how it ended".

This module truncates by *content shape* instead:

* plain text / logs → keep head + tail (8:2), elide the middle;
* JSON → long strings get head/tail truncated, long arrays keep head+tail
  elements (middle elided), objects keep every key and recurse into values
  (only elide entries when the key count is huge). A JSON document embedded in
  a string (e.g. a shell tool's ``{"stdout": "<json>"}``) is parsed and shrunk
  in place.

A final token-accurate backstop (``_text_head_tail``) guarantees the result
fits ``max_tokens`` even when the per-leaf rules alone don't get there.

Only the LLM-facing copy is affected; the displayed/persisted tool output is
left full elsewhere. This module depends on stdlib ``json`` and
``memory.utils`` only (no third-party deps) and must not import ``budgeting``
(would be circular).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from memory.utils import estimate_tokens_in_text, truncate

# Same marker string used by the head-only path in budgeting, kept local to
# avoid a circular import.
TRUNCATED_MARKER = "\n...[content truncated]"

# ~4 chars/token heuristic used when no tokenizer is available (lean mode).
CHARS_PER_TOKEN = 4

# --- tunable thresholds ---------------------------------------------------- #
STRING_MAX_CHARS = 100   # strings longer than this get head/tail truncated
STRING_HEAD_RATIO = 0.8  # 8:2 head:tail split
ARRAY_MAX_ELEMS = 50     # arrays longer than this get their middle elided
ARRAY_HEAD = 30
ARRAY_TAIL = 10
OBJECT_MAX_KEYS = 60     # objects with more keys than this get their middle elided
TEXT_HEAD_RATIO = 0.8    # 8:2 head:tail for the plain-text / backstop path
MAX_DEPTH = 6            # deeper subtrees are collapsed to a truncated dump

_MISSING = object()


def _omit(n: int, unit: str) -> str:
    return f"…[{n} {unit} omitted]…"


def _estimate(text: str, tokenizer: Any) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return max(1, len(text) // CHARS_PER_TOKEN)
    return estimate_tokens_in_text(text, tokenizer=tokenizer)


def _str_head_tail(s: str, budget_chars: int = STRING_MAX_CHARS,
                   ratio: float = STRING_HEAD_RATIO) -> str:
    """Char-based head/tail for a single (JSON leaf) string."""
    if len(s) <= budget_chars:
        return s
    head = int(budget_chars * ratio)
    tail = budget_chars - head
    omitted = len(s) - budget_chars
    return s[:head] + _omit(omitted, "chars") + (s[-tail:] if tail > 0 else "")


def _looks_json(s: str) -> bool:
    t = s.strip()
    return len(t) > STRING_MAX_CHARS and t[:1] in "{[" and t[-1:] in "}]"


def _text_head_tail(text: str, max_tokens: int, tokenizer: Any,
                    ratio: float = TEXT_HEAD_RATIO) -> str:
    """Token-accurate head/tail for a whole string; char fallback if no tokenizer.

    Keeps the first ``max_tokens*ratio`` tokens and the last ``max_tokens*(1-ratio)``
    tokens, with a marker naming how many tokens were dropped from the middle.
    """
    if max_tokens <= 0:
        return ""
    total = _estimate(text, tokenizer)
    if total <= max_tokens:
        return text
    head_tokens = max(1, int(max_tokens * ratio))
    tail_tokens = max(0, max_tokens - head_tokens)
    omitted = max(0, total - max_tokens)

    if tokenizer is None:
        head = text[: head_tokens * CHARS_PER_TOKEN]
        tail = text[-tail_tokens * CHARS_PER_TOKEN:] if tail_tokens > 0 else ""
    else:
        try:
            head = truncate(text, max_token=head_tokens, tokenizer=tokenizer)[0]
            if tail_tokens > 0:
                # `estimate_tokens_in_text` excludes special tokens while
                # `truncate` counts include them, so `total` is a lower bound on
                # the real token count; clamp the window start to stay valid.
                start = max(0, min(total - tail_tokens, total - 1))
                tail = truncate(text, max_token=total, start_token=start,
                                tokenizer=tokenizer)[0]
            else:
                tail = ""
        except Exception:
            head = text[: head_tokens * CHARS_PER_TOKEN]
            tail = text[-tail_tokens * CHARS_PER_TOKEN:] if tail_tokens > 0 else ""
    return f"{head}\n{_omit(omitted, 'tokens')}\n{tail}"


def _shrink(node: Any, depth: int = 0) -> Any:
    """Recursively shrink a parsed-JSON value by the structural rules."""
    if depth >= MAX_DEPTH:
        try:
            dumped = json.dumps(node, ensure_ascii=False)
        except (TypeError, ValueError):
            dumped = str(node)
        return _str_head_tail(dumped)

    if isinstance(node, str):
        if _looks_json(node):
            try:
                parsed = json.loads(node)
            except (ValueError, TypeError):
                parsed = _MISSING
            if parsed is not _MISSING:
                # Re-serialize the shrunk structure back into the string slot,
                # keeping the document's shape (value stays a string).
                return json.dumps(_shrink(parsed, depth + 1), ensure_ascii=False)
        return _str_head_tail(node)

    if isinstance(node, list):
        if len(node) > ARRAY_MAX_ELEMS:
            omitted = len(node) - ARRAY_HEAD - ARRAY_TAIL
            kept = (node[:ARRAY_HEAD]
                    + [_omit(omitted, "elements")]
                    + (node[-ARRAY_TAIL:] if ARRAY_TAIL else []))
        else:
            kept = node
        return [_shrink(x, depth + 1) for x in kept]

    if isinstance(node, dict):
        items = list(node.items())
        if len(items) > OBJECT_MAX_KEYS:
            # Keys are semantic — don't drop them blindly; only when a dict is
            # effectively a huge map do we keep head+tail entries (dicts ordered).
            omitted = len(items) - ARRAY_HEAD - ARRAY_TAIL
            result = {k: _shrink(v, depth + 1) for k, v in items[:ARRAY_HEAD]}
            result["__omitted__"] = _omit(omitted, "keys")
            for k, v in (items[-ARRAY_TAIL:] if ARRAY_TAIL else []):
                result[k] = _shrink(v, depth + 1)
            return result
        return {k: _shrink(v, depth + 1) for k, v in items}

    return node


def smart_truncate(content: str, max_tokens: int, tokenizer: Any = None) -> str:
    """Structurally truncate ``content`` to roughly ``max_tokens``.

    JSON is shrunk by shape (arrays/objects/long strings); anything else gets a
    head+tail text truncation. A token-accurate backstop guarantees the result
    fits even when the per-leaf rules don't reduce enough. Returns the original
    unchanged when it already fits.
    """
    if not content:
        return content
    if _estimate(content, tokenizer) <= max_tokens:
        return content

    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        obj = _MISSING

    if obj is _MISSING:
        return _text_head_tail(content, max_tokens, tokenizer) + TRUNCATED_MARKER

    shrunk = json.dumps(_shrink(obj), ensure_ascii=False)
    if _estimate(shrunk, tokenizer) > max_tokens:
        shrunk = _text_head_tail(shrunk, max_tokens, tokenizer)
    return shrunk + TRUNCATED_MARKER
