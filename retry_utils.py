"""Jittered exponential backoff for the LLM client retry loop.

Why jitter: when several chat() calls hit the same upstream 429 simultaneously
(say, three workers stream-completing the same model at once), if they all
sleep ``base * 2^attempt`` they synchronise into a second thunderclap.
Multiplying by a 0.5-1.5x random factor staggers them.

Why a thread-counter XOR ns clock seed instead of ``random.random()``:
``random`` shares one global state. If anything else in the process calls
``random.seed()`` or pulls heavily from it, our backoff distribution drifts.
A linear-congruential step on ``counter ^ time.time_ns()`` gives a uniform
draw without touching global state and stays correct across threads (the
counter increment is locked).
"""
import threading
import time


_COUNTER_LOCK = threading.Lock()
_COUNTER = 0

# LCG constants from Knuth (well-known, good enough for picking a jitter
# multiplier — not cryptographic, doesn't need to be).
_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_LCG_M = 1 << 64


def _next_uniform():
    """Return a float in [0, 1)."""
    global _COUNTER
    with _COUNTER_LOCK:
        _COUNTER += 1
        seed = _COUNTER ^ time.time_ns()
    seed &= _LCG_M - 1
    seed = (seed * _LCG_A + _LCG_C) & (_LCG_M - 1)
    return seed / _LCG_M


def jittered_backoff(attempt, *, base=1.0, cap=30.0):
    """Return seconds to sleep before retry ``attempt`` (1-based).

    Schedule (with cap=30):
      attempt=1 → [0.5, 1.5)s
      attempt=2 → [1.0, 3.0)s
      attempt=3 → [2.0, 6.0)s
      attempt=4 → [4.0, 12.0)s
      attempt=5 → [8.0, 24.0)s
      attempt≥6 → capped at 30s

    The cap is applied AFTER the jitter multiplier so that we don't
    over-flatten the distribution at high attempt counts.
    """
    if attempt < 1:
        attempt = 1
    raw = base * (2 ** (attempt - 1))
    multiplier = 0.5 + _next_uniform()  # [0.5, 1.5)
    jittered = raw * multiplier
    return min(jittered, cap)
