import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.context import RunVars, Attachment, AgentContext
from agent.message import Message


def test_runvars_autofills_datetime():
    assert RunVars().current_datetime  # non-empty


def test_agent_context_holds_assembly_inputs():
    ctx = AgentContext(
        system_prompt="sys",
        history=[Message("user", "old")],
        current_turn=Message("user", "now"),
        attachments=[Attachment(name="r.pdf", body="text")],
        hints=["use search-file-chunks"],
        tools=None,
        run_vars=RunVars(current_datetime="2026-06-25"),
    )
    assert ctx.current_turn.content == "now"
    assert ctx.attachments[0].name == "r.pdf"
    assert ctx.hints == ["use search-file-chunks"]
