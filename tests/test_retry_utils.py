"""Unit tests for retry_utils.jittered_backoff."""
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retry_utils import jittered_backoff


def test_attempt_1_within_expected_band():
    for _ in range(50):
        d = jittered_backoff(1, base=1.0, cap=30.0)
        assert 0.5 <= d < 1.5


def test_attempt_grows_exponentially():
    # Sample multiple times to dampen jitter; the lower bound of attempt N
    # is base * 2^(N-1) * 0.5, which grows monotonically.
    samples_low = {a: min(jittered_backoff(a, base=1.0, cap=120.0) for _ in range(30))
                   for a in (1, 2, 3, 4)}
    assert samples_low[1] < samples_low[2] < samples_low[3] < samples_low[4]


def test_cap_caps():
    for _ in range(50):
        d = jittered_backoff(20, base=1.0, cap=5.0)
        assert d <= 5.0


def test_attempt_below_one_treated_as_one():
    # attempt=0 / negative shouldn't blow up — clamps to 1.
    d = jittered_backoff(0, base=1.0, cap=30.0)
    assert 0.5 <= d < 1.5
    d = jittered_backoff(-3, base=1.0, cap=30.0)
    assert 0.5 <= d < 1.5


def test_thread_safe_under_concurrency():
    """Many threads pulling jitter values in parallel must not deadlock or
    crash. We don't strictly assert distribution properties here (1000
    samples isn't enough); we just exercise the lock + counter path."""
    results = []
    errors = []

    def worker():
        try:
            for _ in range(100):
                results.append(jittered_backoff(2, base=1.0, cap=30.0))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(results) == 800
    for d in results:
        assert 1.0 <= d < 3.0


def test_distribution_is_not_constant():
    """The whole point of jitter is that two consecutive calls don't
    return the same value. If the seed math is broken (e.g. mod-zero) this
    test fails fast."""
    samples = {jittered_backoff(3, base=1.0, cap=30.0) for _ in range(100)}
    # We wouldn't expect 100 perfectly distinct floats but >50 unique is
    # easy if the LCG is doing its job at all.
    assert len(samples) > 50
