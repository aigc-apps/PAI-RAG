import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.agent import Agent
from agent.context import AgentContext, RunVars, Attachment
from agent.message import Message


def _ctx(**kw):
    base = dict(system_prompt="SYS", history=[], current_turn=Message("user", "学生经验怎么样"),
                attachments=[], hints=[], tools=None, run_vars=RunVars(current_datetime="2026-06-25 18:00"))
    base.update(kw)
    return AgentContext(**base)


def test_system_prompt_is_first_message():
    msgs = Agent.build_messages(_ctx())
    assert msgs[0].role == "system" and msgs[0].content == "SYS"


def test_current_turn_gets_time_prefix():
    msgs = Agent.build_messages(_ctx())
    assert "[System Time: 2026-06-25 18:00]" in msgs[-1].content
    assert "学生经验怎么样" in msgs[-1].content


def test_attachment_text_lands_in_model_input():
    # THE regression lock: attached file content must reach the model.
    msgs = Agent.build_messages(_ctx(attachments=[Attachment(name="resume.pdf", body="3年经验")]))
    text = msgs[-1].content
    assert '<attached_file name="resume.pdf">' in text
    assert "3年经验" in text


def test_hints_appended_after_attachments():
    msgs = Agent.build_messages(_ctx(
        attachments=[Attachment(name="r.pdf", body="x")],
        hints=["对于较长的文件可调用 search-file-chunks"]))
    assert "search-file-chunks" in msgs[-1].content


def test_history_precedes_current_turn():
    msgs = Agent.build_messages(_ctx(history=[Message("user", "old"), Message("assistant", "reply")]))
    roles = [m.role for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]


def test_multimodal_current_turn_prefixes_text_part():
    turn = Message("user", [{"type": "text", "text": "看图"},
                            {"type": "image_url", "image_url": {"url": "http://x"}}])
    msgs = Agent.build_messages(_ctx(current_turn=turn))
    parts = msgs[-1].content
    assert isinstance(parts, list)
    assert "[System Time:" in parts[0]["text"] and "看图" in parts[0]["text"]


def test_attachment_name_with_quote_is_sanitized():
    # A double-quote in the filename must not break the name="..." attribute.
    msgs = Agent.build_messages(_ctx(attachments=[Attachment(name='a"b.pdf', body="body1")]))
    text = msgs[-1].content
    assert 'name="a\'b.pdf"' in text
    assert "body1" in text
