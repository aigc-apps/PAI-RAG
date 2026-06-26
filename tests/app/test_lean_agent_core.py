# tests/app/test_lean_agent_core.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))


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


def test_budgeting_without_tokenizer(monkeypatch):
    import agent.budgeting as b
    monkeypatch.setattr(b, "get_tokenizer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no tokenizer")))
    mgr = b.AgentMessageManager(context_window=110000, max_output_tokens=8000)
    # estimate still returns a positive int via the length fallback
    n = mgr.estimate_msg_tokens({"role": "user", "content": "hello world this is some text"})
    assert isinstance(n, int) and n > 0
