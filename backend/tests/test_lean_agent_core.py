# ruff: noqa: E401
# tests/app/test_lean_agent_core.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_agent_core_imports_without_trace(monkeypatch):
    # Simulate the trace extension / opentelemetry being absent.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("extensions.trace") or name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError(f"simulated-absent: {name}")
        return real_import(name, *a, **k)

    for m in list(sys.modules):
        if m.startswith("agent.agent") or m.startswith("agent.tools") or m == "agent.budgeting":
            sys.modules.pop(m, None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    import importlib
    import agent.agent as agent_mod  # must import despite missing trace deps
    importlib.reload(agent_mod)
    assert agent_mod.Agent is not None


def test_budgeting_has_no_tokenizer_dependency():
    import agent.budgeting as b
    assert not hasattr(b, "get_tokenizer")
    mgr = b.AgentMessageManager(context_window=110000, max_output_tokens=8000)
    assert not hasattr(mgr, "tokenizer")
    n = mgr.estimate_msg_tokens({"role": "user", "content": "hello world this is some text"})
    assert isinstance(n, int) and n > 0


def test_cap_tool_result_structural_without_tokenizer():
    import json as _json
    import agent.budgeting as b
    mgr = b.AgentMessageManager(context_window=110000, max_output_tokens=8000)
    mgr.max_tool_result_tokens = 100  # force the truncation path

    payload = _json.dumps({
        "exit_code": 0,
        "rows": [{"i": i, "blob": "q" * 200} for i in range(300)],
        "TotalCount": 300,
    })
    out = mgr.cap_tool_result(payload)
    assert out != payload
    assert out.endswith(b.TOOL_RESULT_TRUNCATED_MARKER)   # structural path reuses the marker
    assert len(out) < len(payload)                        # far smaller
    assert "TotalCount" in out                            # tail survives (head-only would drop it)

    # content already under the cap is returned untouched
    assert mgr.cap_tool_result("tiny result") == "tiny result"
