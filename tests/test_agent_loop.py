import unittest

from agent_loop import BaseHandler, agent_runner_loop
from llm_client import Response


class FakeClient:
    def __init__(self, content):
        self.content = content

    def chat(self, system, new_messages, tools):
        yield self.content
        return Response(content=self.content)


class AgentLoopEventTests(unittest.TestCase):
    def run_loop_events(self, content):
        events = []
        exit_reason = agent_runner_loop(
            client=FakeClient(content),
            system_prompt="system",
            user_input="user",
            handler=BaseHandler(),
            tools_schema=[],
            max_turns=1,
            on_event=events.append,
        )
        return events, exit_reason

    def test_no_tool_call_emits_final_answer(self):
        events, exit_reason = self.run_loop_events("final answer")

        chunks = [
            event["content"]["text"]
            for event in events
            if event.get("sessionUpdate") == "agent_message_chunk"
        ]
        self.assertEqual(chunks, ["final answer"])
        self.assertEqual(exit_reason["result"], "NO_TOOL_CALL")

    def test_thinking_only_no_tool_call_emits_visible_fallback(self):
        events, exit_reason = self.run_loop_events("<thinking>service is running</thinking>")

        chunks = [
            event["content"]["text"]
            for event in events
            if event.get("sessionUpdate") == "agent_message_chunk"
        ]
        self.assertEqual(chunks, ["service is running"])
        self.assertEqual(exit_reason["result"], "NO_TOOL_CALL")


if __name__ == "__main__":
    unittest.main()
