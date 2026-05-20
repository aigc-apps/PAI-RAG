"""Translate any upstream exception into retry / rotate / compress decisions.

llm_client.chat() catches exceptions and calls ``classify_api_error(exc)`` to get
a ``ClassifiedError``; it then reads three booleans (retryable /
should_rotate_credential / should_compress) and acts. This keeps the retry loop
in chat() free of provider-specific branching.

What this module knows that a status-code switch doesn't:

- Status codes can hide several layers deep inside ``__cause__`` /
  ``__context__`` chains (typical for SDKs that re-raise their own error type
  on top of httpx errors). We walk both chains, depth-limited, to surface them.
- OpenRouter wraps the real upstream error inside ``error.metadata.raw`` as a
  JSON string; the outer ``error.code`` is OpenRouter's own re-encoding. We
  unwrap once so a real 429 from Anthropic doesn't masquerade as a 200/400.
- Some failures have no status code at all (network errors, parse errors,
  truncated streams). Keyword matching on ``str(exc)`` catches the common
  shapes (context overflow, overloaded, timeout) so we still make the right
  call instead of defaulting to "give up immediately".
"""
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FailureReason(Enum):
    AUTH = 'auth'
    AUTH_PERMANENT = 'auth_permanent'
    RATE_LIMIT = 'rate_limit'
    OVERLOADED = 'overloaded'
    SERVER_ERROR = 'server_error'
    TIMEOUT = 'timeout'
    CONTEXT_OVERFLOW = 'context_overflow'
    BAD_REQUEST = 'bad_request'
    UNKNOWN = 'unknown'


@dataclass
class ClassifiedError:
    reason: FailureReason
    status_code: Optional[int] = None
    message: str = ''
    retryable: bool = True
    should_compress: bool = False
    should_rotate_credential: bool = False


# ────────────── Cause-chain walk ────────────── #

_MAX_CAUSE_DEPTH = 5


def _walk_causes(exc):
    """Yield exc and its __cause__/__context__ chain, deduped, up to a depth cap.

    BaseException's chain links can technically loop (rare but real — anyone
    who re-raises inside an except can make it happen). The seen-id set keeps
    us from spinning, the depth cap keeps us from staring into a 50-deep
    SDK→httpx→ssl onion."""
    seen = set()
    stack = [(exc, 0)]
    while stack:
        cur, depth = stack.pop()
        if cur is None or id(cur) in seen or depth > _MAX_CAUSE_DEPTH:
            continue
        seen.add(id(cur))
        yield cur
        cause = getattr(cur, '__cause__', None)
        ctx = getattr(cur, '__context__', None)
        if cause is not None:
            stack.append((cause, depth + 1))
        if ctx is not None and ctx is not cause:
            stack.append((ctx, depth + 1))


def _extract_status_code(exc):
    """Best-effort find an int status code anywhere in the cause chain.

    Checks ``.status_code`` directly, then ``.response.status_code``. Returns
    the first hit. None if nothing in the chain has one."""
    for layer in _walk_causes(exc):
        code = getattr(layer, 'status_code', None)
        if code is None:
            resp = getattr(layer, 'response', None)
            if resp is not None:
                code = getattr(resp, 'status_code', None)
        if code is None:
            continue
        try:
            return int(code)
        except (TypeError, ValueError):
            continue
    return None


def _extract_body_text(exc):
    """Pull a body / message string out of any layer that has one.

    Order of preference: ``.body`` (OpenAI SDK stashes the raw json string
    here), ``.response.text``, then ``str(exc)``. Concatenating them is fine —
    keyword matching downstream tolerates noise."""
    parts = []
    for layer in _walk_causes(exc):
        body = getattr(layer, 'body', None)
        if isinstance(body, str) and body:
            parts.append(body)
        elif isinstance(body, (dict, list)):
            try:
                parts.append(json.dumps(body, ensure_ascii=False))
            except Exception:
                pass
        resp = getattr(layer, 'response', None)
        if resp is not None:
            text = getattr(resp, 'text', None)
            if isinstance(text, str) and text:
                parts.append(text)
        msg = getattr(layer, 'message', None)
        if isinstance(msg, str) and msg:
            parts.append(msg)
    parts.append(str(exc))
    return '\n'.join(parts)


# ────────────── OpenRouter unwrap ────────────── #

def _unwrap_openrouter(body_text):
    """Try to find a real upstream status/message hidden inside OR's wrap.

    OpenRouter's error envelope looks like::

        {"error": {"code": 429, "metadata": {"raw": "{\\"error\\":{\\"code\\":529,...}}"}}}

    The outer code is OR's transcoding of whatever the upstream said; the inner
    JSON in ``metadata.raw`` is the truth. We parse one level deep — that
    covers every shape we've seen in the wild without making this routine
    grow unbounded.

    Returns ``(status_code, message)`` from the inner error if found, else
    ``(None, None)``."""
    if not isinstance(body_text, str):
        return None, None
    # Body might be embedded inside other text — find the first {...} block
    # that mentions metadata.raw and try to parse from there.
    if 'metadata' not in body_text or 'raw' not in body_text:
        return None, None
    # Try a few candidate start positions: each '{' that looks like the start
    # of a JSON object.
    for start in (m.start() for m in re.finditer(r'\{', body_text)):
        try:
            obj, _ = json.JSONDecoder().raw_decode(body_text[start:])
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(obj, dict):
            continue
        err = obj.get('error') if isinstance(obj.get('error'), dict) else obj
        if not isinstance(err, dict):
            continue
        meta = err.get('metadata') if isinstance(err.get('metadata'), dict) else None
        raw = meta.get('raw') if meta else None
        if not isinstance(raw, str):
            continue
        try:
            inner = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(inner, dict):
            continue
        inner_err = inner.get('error') if isinstance(inner.get('error'), dict) else inner
        if not isinstance(inner_err, dict):
            continue
        code = inner_err.get('code') or inner_err.get('status') or inner_err.get('status_code')
        try:
            code_int = int(code) if code is not None else None
        except (TypeError, ValueError):
            code_int = None
        msg = inner_err.get('message') or ''
        if isinstance(msg, str) and (code_int or msg):
            return code_int, msg
    return None, None


# ────────────── Text-pattern fallback ────────────── #

_CONTEXT_OVERFLOW_PATTERNS = (
    'context length',
    'maximum context',
    'context_length_exceeded',
    'token limit',
    'too many tokens',
    'prompt is too long',
    'reduce the length',
)
_OVERLOADED_PATTERNS = ('overloaded', 'capacity', 'unavailable')
_TIMEOUT_PATTERNS = ('timeout', 'timed out', 'read timed out')
_RATE_LIMIT_PATTERNS = ('rate limit', 'too many requests', 'quota exceeded')


def _text_match_any(text, patterns):
    if not isinstance(text, str):
        return False
    low = text.lower()
    return any(p in low for p in patterns)


# ────────────── Status-code → reason ────────────── #

def _reason_from_status(code, text):
    """Map an HTTP status to a FailureReason. ``text`` lets us split 400 into
    BAD_REQUEST vs CONTEXT_OVERFLOW (both are 400, semantically very different)."""
    if code is None:
        return None
    if code == 401:
        return FailureReason.AUTH
    if code == 403:
        return FailureReason.AUTH_PERMANENT
    if code == 429:
        return FailureReason.RATE_LIMIT
    if code in (503, 529):
        return FailureReason.OVERLOADED
    if 500 <= code < 600:
        return FailureReason.SERVER_ERROR
    if code == 408:
        return FailureReason.TIMEOUT
    if code == 400:
        if _text_match_any(text, _CONTEXT_OVERFLOW_PATTERNS):
            return FailureReason.CONTEXT_OVERFLOW
        return FailureReason.BAD_REQUEST
    if 400 <= code < 500:
        return FailureReason.BAD_REQUEST
    return None


# ────────────── Decision matrix ────────────── #

_DECISION = {
    FailureReason.AUTH:             dict(retryable=True,  rotate=True,  compress=False),
    FailureReason.AUTH_PERMANENT:   dict(retryable=False, rotate=True,  compress=False),
    FailureReason.RATE_LIMIT:       dict(retryable=True,  rotate=True,  compress=False),
    FailureReason.OVERLOADED:       dict(retryable=True,  rotate=False, compress=False),
    FailureReason.SERVER_ERROR:     dict(retryable=True,  rotate=False, compress=False),
    FailureReason.TIMEOUT:          dict(retryable=True,  rotate=False, compress=False),
    FailureReason.CONTEXT_OVERFLOW: dict(retryable=True,  rotate=False, compress=True),
    FailureReason.BAD_REQUEST:      dict(retryable=False, rotate=False, compress=False),
    FailureReason.UNKNOWN:          dict(retryable=True,  rotate=False, compress=False),
}


def classify_api_error(exc):
    """Translate any exception into a ``ClassifiedError``.

    Decision flow:
      1. Walk cause chain to find any status code
      2. Pull together a body/message text from every layer
      3. Try OpenRouter unwrap — if its inner status disagrees with the outer,
         the inner wins (it's the truthful one)
      4. Map (status, text) → FailureReason; if no status, fall back to text
         keyword matching; if still nothing, UNKNOWN
      5. Decision matrix gives retryable / rotate / compress booleans
    """
    if exc is None:
        return ClassifiedError(reason=FailureReason.UNKNOWN, message='')

    status = _extract_status_code(exc)
    body_text = _extract_body_text(exc)

    inner_code, inner_msg = _unwrap_openrouter(body_text)
    if inner_code is not None:
        # Inner wins. Replace status; keep both messages so downstream logs
        # know we unwrapped.
        status = inner_code
        if inner_msg:
            body_text = body_text + '\n' + inner_msg

    reason = _reason_from_status(status, body_text)
    if reason is None:
        if _text_match_any(body_text, _CONTEXT_OVERFLOW_PATTERNS):
            reason = FailureReason.CONTEXT_OVERFLOW
        elif _text_match_any(body_text, _RATE_LIMIT_PATTERNS):
            reason = FailureReason.RATE_LIMIT
        elif _text_match_any(body_text, _OVERLOADED_PATTERNS):
            reason = FailureReason.OVERLOADED
        elif _text_match_any(body_text, _TIMEOUT_PATTERNS):
            reason = FailureReason.TIMEOUT
        else:
            reason = FailureReason.UNKNOWN

    decision = _DECISION[reason]
    short_msg = (body_text[:300] + '…') if len(body_text) > 300 else body_text
    return ClassifiedError(
        reason=reason,
        status_code=status,
        message=short_msg,
        retryable=decision['retryable'],
        should_rotate_credential=decision['rotate'],
        should_compress=decision['compress'],
    )
