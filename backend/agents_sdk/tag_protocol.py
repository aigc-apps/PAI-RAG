"""Shared classification for XML-style protocol tags emitted by the model.

Three layers (streaming filter, session replay, legacy ACP path) classify
``<tag>`` openers in model output. Keeping the classification function and
the DISCARD list here is the single source of truth — unknown tags default
to ``'thought'`` so newly invented protocol tags can never leak to the
user-visible text channel.
"""
from __future__ import annotations

import re

DISCARD_TAG_NAMES: frozenset[str] = frozenset({'summary', 'forcing_skill_activation'})
# System markers the agent layer intentionally emits to the user. These must
# round-trip verbatim through every stripping pass, otherwise the persisted
# message UI loses the wrapper that tells users a tool output was elided to
# disk (``<persisted-output>`` + ``<preview>``) or that a chunk is the
# verbatim content of an assistant-written file (``<file_content>``).
VISIBLE_HTML_TAG_NAMES: frozenset[str] = frozenset({
    'persisted-output',
    'preview',
    'file_content',
})

TAG_NAME_RE = re.compile(r'[A-Za-z][\w-]*')


def classify_tag(name: str) -> str:
    n = name.lower()
    if n in DISCARD_TAG_NAMES:
        return 'discard'
    if n in VISIBLE_HTML_TAG_NAMES:
        return 'visible'
    return 'thought'
