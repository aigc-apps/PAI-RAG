"""Tiered context management — the offload-not-truncate primitive.

When the token budget forces an old tool result out of the model window, we do NOT
throw its bytes away (the old lossy `_truncate` + "[content truncated]" marker).
Instead we *offload* it: the window keeps a compact placeholder carrying a handle,
and the full body stays recoverable. The model can pull it back verbatim with the
``read_handle`` tool. This is Anthropic's "context editing / clear tool uses"
applied to a distributed, server-side loop:

- Source of truth is the durable store (``function_call_output`` items hold the full
  output across runs). The window is a re-derivable *projection* — offloading only
  edits the projection, never the store, so a stateless worker rebuilding from
  ``conversation_id`` still has every byte.
- Within a single run a freshly-produced result is not persisted yet (persistence
  happens at run end). So the offloader also stashes the full body in a per-run,
  in-memory dict keyed by ``call_id``; ``read_handle`` checks that dict first, then
  falls back to the store for older runs. No worker-local disk, no file writes.

The per-run dict is owned by ``Agent.run``: it is handed to the budget manager (which
writes bodies as it offloads) and carried on the ``ToolScope`` (so ``read_handle``
reads it). Explicit passing — not a ContextVar — because the run's event generator
is driven across task/context boundaries (background streaming, resume), where a
``ContextVar.reset(token)`` would fail ("created in a different Context"). A fresh
Agent (hence fresh dict) per run also isolates nested subagent runs automatically.
"""
from __future__ import annotations

from utils.constants import try_get_int_env

# A tool result larger than this (in estimated tokens) is eligible to be offloaded
# to a placeholder when the budget compressor reaches it. ~4k tokens ≈ 16KB of
# text; sits at the current per-result cap so nothing smaller ever gets a handle.
OFFLOAD_SOFT_TOKENS = try_get_int_env("OFFLOAD_SOFT_TOKENS", 4000)
# How much of the head to keep inline in the placeholder, so the model can still
# judge relevance (and often answer) without a read_handle round-trip.
OFFLOAD_DIGEST_CHARS = try_get_int_env("OFFLOAD_DIGEST_CHARS", 400)

# Handles are opaque-ish but human-legible: store://tool/<call_id>. read_handle
# also accepts a bare call_id, so the model can't fumble the scheme.
_HANDLE_SCHEME = "store://tool/"
# A stable first-line sentinel so the compressor never re-offloads an already
# offloaded result, and builder/tests can recognise a placeholder.
OFFLOAD_SENTINEL = "[offloaded tool result]"


# --- Handles + placeholders ---------------------------------------------------
def handle_for(call_id: str) -> str:
    return f"{_HANDLE_SCHEME}{call_id}"


def parse_handle(handle: str) -> str:
    """Extract the ``call_id`` from a handle. Accepts the full ``store://tool/<id>``
    form or a bare id; returns ``""`` for anything unusable."""
    if not handle:
        return ""
    h = handle.strip()
    if h.startswith(_HANDLE_SCHEME):
        return h[len(_HANDLE_SCHEME):].strip()
    # Tolerate a bare call_id (what a model is most likely to echo back).
    return h


def is_placeholder(content: str) -> bool:
    return isinstance(content, str) and content.startswith(OFFLOAD_SENTINEL)


def make_placeholder(call_id: str, full: str, tokens: int) -> str:
    """Build the compact window replacement for an offloaded tool result: a header
    naming the handle + size, then a head digest so the model keeps some signal."""
    digest = (full or "")[:OFFLOAD_DIGEST_CHARS]
    truncated = len(full or "") > OFFLOAD_DIGEST_CHARS
    header = (
        f"{OFFLOAD_SENTINEL} ~{tokens} tokens, preserved in full. "
        f"Call read_handle(handle=\"{handle_for(call_id)}\") to fetch the complete "
        f"output (optional line range via start/count)."
    )
    if not digest:
        return header
    tail = " …(truncated; use read_handle for the rest)" if truncated else ""
    return f"{header}\n--- head preview ---\n{digest}{tail}"
