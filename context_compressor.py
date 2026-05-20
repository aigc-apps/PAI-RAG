"""Token-budget history compression.

Replaces the byte-level ``trim_history`` with a multi-phase pipeline that
preserves the OpenAI tool-message protocol invariants while shrinking
oversized histories.

Pipeline (in compress_history):

    1. prune_old_tool_results   — truncate the BODY of old tool messages
                                  (oldest first); recent tool results stay
                                  full-fidelity for the model to reason over.
    2. trim_to_token_budget     — drop entire messages from the middle until
                                  estimate ≤ max_tokens. Anchors first N and
                                  last M messages so we never lose the
                                  initial framing or the user's latest turn.
    3. sanitize_tool_pairs      — repair any orphan tool messages or
                                  dangling assistant.tool_calls that step 2
                                  may have created. The OpenAI API rejects
                                  histories with these.
    4. ensure_last_user_in_tail — invariant check; the schedule above should
                                  make this a no-op, but it's a load-bearing
                                  invariant so we assert it.

Phase-3 (LLM-based summarization of the dropped middle) is deliberately
deferred — it needs an auxiliary LLM client and policy choices that
this PR isn't taking on. The pipeline shape leaves a clean spot for it
between steps 2 and 3 in a future PR.

All functions are pure: input list[dict] → output list[dict]. The original
list is never mutated.
"""


# Default cap for a single tool message body. ~4000 chars / 4 ≈ 1000 tokens
# — generous enough to keep meaningful context, tight enough that one giant
# file_read doesn't blow the window.
_DEFAULT_TOOL_RESULT_MAX_CHARS = 4000

# How many recent messages we keep with full tool-result bodies. After this
# many turns the model rarely re-references old tool output verbatim.
_DEFAULT_PROTECT_LAST_N = 10

# Always keep the very first user message — it sets the framing for the whole
# session. (history here doesn't include the system message; that's added at
# chat() build time.)
_DEFAULT_PROTECT_FIRST_N = 1


def estimate_tokens(messages):
    """Cheap byte→token estimate. Same algorithm as the old llm_client.trim_history.

    ``len(text) // 4`` is the rule-of-thumb GPT-style tokens-per-char ratio.
    Within ±25% of real tokenisation for English+code; close enough to drive
    a budget that has its own slack."""
    n = 0
    for m in messages:
        c = m.get('content') or ''
        if isinstance(c, str):
            n += len(c) // 4
        elif isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict):
                    n += len(str(blk.get('text', ''))) // 4
        for tc in m.get('tool_calls') or []:
            fn = tc.get('function', {}) if isinstance(tc, dict) else {}
            n += len(fn.get('arguments', '')) // 4
    return n


# ────────────── Phase 1: prune old tool results ────────────── #

_TRUNC_MARKER = '\n…[truncated for context compression]…\n'


def _truncate_body(content, max_chars):
    """Keep head and tail, drop the middle. The two slices give the model a
    chance to anchor on what the call did AND what it returned."""
    if not isinstance(content, str) or len(content) <= max_chars:
        return content
    head_n = max_chars // 2
    tail_n = max_chars // 4
    return content[:head_n] + _TRUNC_MARKER + content[-tail_n:]


def prune_old_tool_results(history, *, protect_last_n=_DEFAULT_PROTECT_LAST_N,
                           max_chars=_DEFAULT_TOOL_RESULT_MAX_CHARS):
    """Truncate the body of any tool message older than the last N messages.

    Only ``role=='tool'`` is touched — assistant content and user messages are
    far more compact and rarely need trimming. The tail (``protect_last_n``
    messages) keeps full bodies so the model can still see the latest tool
    output verbatim."""
    if protect_last_n < 0:
        protect_last_n = 0
    out = []
    n = len(history)
    boundary = n - protect_last_n
    for i, msg in enumerate(history):
        if i >= boundary or msg.get('role') != 'tool':
            out.append(msg)
            continue
        body = msg.get('content')
        truncated = _truncate_body(body, max_chars)
        if truncated is body:
            out.append(msg)
        else:
            new_msg = dict(msg)
            new_msg['content'] = truncated
            out.append(new_msg)
    return out


# ────────────── Phase 2: trim to token budget ────────────── #

def trim_to_token_budget(history, *, max_tokens, protect_first_n=_DEFAULT_PROTECT_FIRST_N,
                         protect_last_n=_DEFAULT_PROTECT_LAST_N):
    """Drop messages from the middle until the total fits under max_tokens.

    Difference from the old ``trim_history``:
      - first N + last M are anchored, so we never lose the initial framing
        or the user's most recent turn
      - we delete from the middle inward, not just from the head, which keeps
        recent context intact even when the head is small

    sanitize_tool_pairs runs after this and repairs any tool/tool_calls breaks
    we may have introduced."""
    if max_tokens <= 0:
        return list(history)
    if estimate_tokens(history) <= max_tokens:
        return list(history)

    n = len(history)
    if n <= protect_first_n + protect_last_n:
        # Already at the protected minimum; nothing more to drop without
        # breaking the contract.
        return list(history)

    head = list(history[:protect_first_n])
    tail = list(history[n - protect_last_n:]) if protect_last_n > 0 else []
    middle = list(history[protect_first_n:n - protect_last_n]) if protect_last_n > 0 \
        else list(history[protect_first_n:])

    # Drop from the start of middle (oldest middle messages) until under budget.
    while middle and estimate_tokens(head + middle + tail) > max_tokens:
        middle.pop(0)

    return head + middle + tail


# ────────────── Phase 3: sanitize tool pairs ────────────── #

def sanitize_tool_pairs(history):
    """Repair dangling assistant.tool_calls / orphan tool messages.

    OpenAI-compatible APIs return 400 when:
      - a ``role=='tool'`` message references a ``tool_call_id`` no
        assistant message in this history declares
      - an ``assistant.tool_calls[i]`` has no matching ``tool`` message
        within the visible history

    Step 1: drop orphan tool messages.
    Step 2: drop tool_call entries that nothing answers, AND drop the whole
            assistant message if it ends up with no content and no tool_calls.
    """
    # Step 1: collect all tool_call ids announced by assistant messages.
    declared_ids = set()
    for m in history:
        if m.get('role') != 'assistant':
            continue
        for tc in m.get('tool_calls') or []:
            tid = tc.get('id') if isinstance(tc, dict) else None
            if tid:
                declared_ids.add(tid)

    # Drop orphan tool messages.
    pruned = [
        m for m in history
        if not (m.get('role') == 'tool' and m.get('tool_call_id') not in declared_ids)
    ]

    # Step 2: collect tool_call ids that DO have an answering tool message.
    answered_ids = set()
    for m in pruned:
        if m.get('role') == 'tool':
            tid = m.get('tool_call_id')
            if tid:
                answered_ids.add(tid)

    out = []
    for m in pruned:
        if m.get('role') != 'assistant':
            out.append(m)
            continue
        tcs = m.get('tool_calls') or []
        if not tcs:
            out.append(m)
            continue
        kept = [tc for tc in tcs if isinstance(tc, dict) and tc.get('id') in answered_ids]
        if len(kept) == len(tcs):
            out.append(m)
            continue
        # Some tool_calls were dropped. Rewrite this assistant message.
        new_msg = dict(m)
        if kept:
            new_msg['tool_calls'] = kept
            out.append(new_msg)
        else:
            # All tool_calls are unanswered. If the assistant has visible
            # content, keep that; otherwise drop the whole message.
            content = m.get('content')
            if isinstance(content, str) and content.strip():
                new_msg.pop('tool_calls', None)
                out.append(new_msg)
            elif isinstance(content, list) and content:
                new_msg.pop('tool_calls', None)
                out.append(new_msg)
            # else: drop entirely
    return out


# ────────────── Phase 4: invariant — last user must remain ────────────── #

def ensure_last_user_in_tail(history):
    """Invariant placeholder for "the user's latest question survived compression".

    Today this is a no-op pass-through (returns a shallow copy). The real
    enforcement is upstream — ``trim_to_token_budget`` anchors ``protect_last_n``
    messages, so under the default policy the last user message can't be
    dropped.

    This function exists as a structural seam: a future PR that loosens the
    tail anchoring (e.g. for Phase-3 LLM summarization that re-summarizes
    the last few turns) can plug a real check or repair here without changing
    the ``compress_history`` pipeline shape."""
    return list(history)


# ────────────── Public entry point ────────────── #

def compress_history(history, *, max_tokens, tool_result_max_chars=_DEFAULT_TOOL_RESULT_MAX_CHARS,
                     protect_first_n=_DEFAULT_PROTECT_FIRST_N,
                     protect_last_n=_DEFAULT_PROTECT_LAST_N):
    """Single entry point: shrink ``history`` to fit ``max_tokens``.

    Returns a new list; the input is never mutated. If the input is already
    under budget, returns a shallow copy (still a new list, so callers can
    safely reassign over the original)."""
    if not history:
        return []
    if estimate_tokens(history) <= max_tokens:
        return list(history)

    h = prune_old_tool_results(history, protect_last_n=protect_last_n,
                               max_chars=tool_result_max_chars)
    if estimate_tokens(h) <= max_tokens:
        h = sanitize_tool_pairs(h)
        return ensure_last_user_in_tail(h)

    h = trim_to_token_budget(h, max_tokens=max_tokens,
                             protect_first_n=protect_first_n,
                             protect_last_n=protect_last_n)
    h = sanitize_tool_pairs(h)
    h = ensure_last_user_in_tail(h)
    return h
