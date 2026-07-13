# ruff: noqa: E401, E402
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agent.budgeting as b
from agent.budgeting import AgentMessageManager, approx_tokens


def _mgr():
    # These tests exercise the character-estimate fast path and cache wiring.
    return AgentMessageManager(context_window=110000, max_output_tokens=8000)


# --------------------------------------------------------------------------- #
# approx_tokens: cheap, script-aware, and SAFE (over-counts) for CJK
# --------------------------------------------------------------------------- #
def test_approx_tokens_empty_and_positive():
    assert approx_tokens("") == 0
    assert approx_tokens("hello world") > 0


def test_approx_tokens_chinese_not_undercounted():
    # The whole reason for a script-aware estimate: a naive chars/4 counts 7 Han
    # chars as 1 token, which would wave an over-budget CJK context through the
    # gate. approx_tokens must count each CJK char close to ~1 token.
    zh = "智能体调用工具"  # 7 Han chars
    assert approx_tokens(zh) > len(zh) // 4
    assert approx_tokens(zh) >= 4


def test_approx_tokens_cjk_heavier_per_char_than_latin():
    # Per character, CJK must cost more tokens than Latin prose.
    zh = "数据库连接池预热" * 20
    en = "database connection pool warmup" * 20
    assert approx_tokens(zh) / len(zh) > approx_tokens(en) / len(en)


# --------------------------------------------------------------------------- #
# fast-path gate: exact tokenization is skipped when comfortably under budget
# --------------------------------------------------------------------------- #
def test_fast_path_skips_exact_when_under_budget():
    mgr = _mgr()
    calls = {"n": 0}
    orig = mgr.estimate_messages_tokens
    mgr.estimate_messages_tokens = lambda m: (calls.__setitem__("n", calls["n"] + 1) or orig(m))

    msgs = [{"role": "user", "content": "a short question"}]
    out = mgr.fit_to_budget(msgs)

    assert out is msgs             # returned unchanged
    assert calls["n"] == 0         # exact accounting never ran


def test_over_budget_falls_through_to_exact_path():
    mgr = _mgr()
    mgr.token_budget = 50          # tiny budget forces the gate to fall through
    calls = {"n": 0}
    orig = mgr.estimate_messages_tokens
    mgr.estimate_messages_tokens = lambda m: (calls.__setitem__("n", calls["n"] + 1) or orig(m))

    mgr.fit_to_budget([{"role": "user", "content": "word " * 500}])
    assert calls["n"] >= 1         # gate did NOT early-return; exact accounting ran


# --------------------------------------------------------------------------- #
# per-message count cache: a message is tokenized once per turn, not per step
# --------------------------------------------------------------------------- #
def test_estimate_msg_tokens_is_cached_by_content():
    mgr = _mgr()
    calls = {"n": 0}
    orig = mgr._estimate_msg_tokens_uncached
    mgr._estimate_msg_tokens_uncached = lambda m: (calls.__setitem__("n", calls["n"] + 1) or orig(m))

    msg = {"role": "tool", "content": "some tool output that repeats across steps"}
    first = mgr.estimate_msg_tokens(msg)
    second = mgr.estimate_msg_tokens(dict(msg))   # fresh dict, same content → cache hit

    assert first == second
    assert calls["n"] == 1                         # uncached body ran exactly once


# --------------------------------------------------------------------------- #
# cap_tool_result: length is a provable upper bound, so small results skip the
# estimator entirely; oversized ones still truncate.
# --------------------------------------------------------------------------- #
def test_cap_tool_result_small_never_tokenizes(monkeypatch):
    mgr = _mgr()
    mgr.max_tool_result_tokens = 100
    monkeypatch.setattr(
        b, "_estimate_tokens",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("estimator must not run")),
    )
    small = "x" * 100                              # 100 chars ≤ cap ⇒ tokens ≤ 100
    assert mgr.cap_tool_result(small) == small


def test_cap_tool_result_large_still_truncates():
    import json as _json
    mgr = _mgr()
    mgr.max_tool_result_tokens = 100
    payload = _json.dumps({"rows": [{"i": i, "blob": "q" * 200} for i in range(300)]})
    out = mgr.cap_tool_result(payload)
    assert out != payload
    assert out.endswith(b.TOOL_RESULT_TRUNCATED_MARKER)
    assert len(out) < len(payload)
