from agent.agent import Agent
from agent.context import AgentContext, RunVars
from agent.message import Message
from agent.tools.base import ToolBox


def _context(*, current_turn=None, context_block="", tools=None):
    return AgentContext(
        system_prompt="# Persona\nBe helpful.",
        history=[
            Message(role="user", content="earlier question"),
            Message(role="assistant", content="earlier answer"),
        ],
        current_turn=current_turn or Message(role="user", content="current question"),
        attachments=[],
        hints=[],
        tools=tools or ToolBox([]),
        run_vars=RunVars(current_date="2026-07-14", timezone="Asia/Shanghai"),
        context_block=context_block,
    )


def test_build_messages_merges_runtime_context_into_single_system_message():
    messages = Agent.build_messages(
        _context(context_block="# Memory\nThe user prefers concise answers.")
    )

    system_messages = [message for message in messages if message.role == "system"]
    assert len(system_messages) == 1
    system = system_messages[0].content
    assert system.startswith("# Persona\nBe helpful.")
    assert "# Environment" in system
    assert "Today's date: 2026-07-14" in system
    assert "Time zone: Asia/Shanghai" in system
    assert "12:34:56" not in system
    assert "# Memory\nThe user prefers concise answers." in system
    assert system.index("# Environment") < system.index("# Memory")

    assert [(message.role, message.content) for message in messages[1:]] == [
        ("user", "earlier question"),
        ("assistant", "earlier answer"),
        ("user", "current question"),
    ]
    assert all("<system-reminder>" not in str(message.content) for message in messages)


def test_build_messages_omits_blank_runtime_context():
    messages = Agent.build_messages(_context(context_block="  \n"))

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0].content.endswith("Time zone: Asia/Shanghai")


def test_build_messages_points_to_datetime_tool_only_when_available():
    async def now():
        return "2026-07-14 12:34:56"

    from agent.tools.base import Tool

    datetime_tool = Tool(
        name="current_datetime",
        description="time",
        parameters={"type": "object", "properties": {}},
        fn=now,
    )

    with_tool = Agent.build_messages(_context(tools=ToolBox([datetime_tool])))[0]
    without_tool = Agent.build_messages(_context())[0]

    assert "For the exact current date or time, call current_datetime." in with_tool.content
    assert "call current_datetime" not in without_tool.content


def test_build_messages_keeps_multimodal_user_content_free_of_system_time():
    turn = Message(
        role="user",
        content=[
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ],
    )

    messages = Agent.build_messages(_context(current_turn=turn))

    assert len([message for message in messages if message.role == "system"]) == 1
    assert messages[-1].content[0]["text"] == "describe this"
    assert "System Time" not in messages[-1].content[0]["text"]
