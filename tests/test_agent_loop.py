import os
import tempfile
import unittest

from agent_loop import BaseHandler, StepOutcome, agent_runner_loop
from llm_client import Response, ToolCall


class FakeClient:
    def __init__(self, content):
        self.content = content

    def chat(self, system, new_messages, tools):
        yield self.content
        return Response(content=self.content)


class SequenceFakeClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.new_messages = []

    def chat(self, system, new_messages, tools):
        self.new_messages.append(new_messages)
        content = self.contents.pop(0)
        yield content
        return Response(content=content)


class ToolThenFinalClient:
    def __init__(self, tool_calls, final_content="done"):
        self.tool_calls = tool_calls
        self.final_content = final_content
        self.new_messages = []

    def chat(self, system, new_messages, tools):
        self.new_messages.append(new_messages)
        if len(self.new_messages) == 1:
            if False:
                yield ""
            return Response(content="", tool_calls=self.tool_calls)
        yield self.final_content
        return Response(content=self.final_content)


class BigToolHandler(BaseHandler):
    def __init__(self, output_root):
        self.output_root = output_root

    def do_big_tool(self, args, response):
        return StepOutcome("x" * 120000, next_prompt="continue")

    def persist_tool_result(self, content, tool_call_id, subdir="tool_results"):
        directory = os.path.join(self.output_root, subdir)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{tool_call_id}.txt")
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        return path


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

    def test_summary_only_retries_for_visible_final_answer(self):
        client = SequenceFakeClient([
            "<summary>confirmed root cause</summary>",
            "诊断结论: scene home_feed19 is not configured",
        ])
        events = []

        exit_reason = agent_runner_loop(
            client=client,
            system_prompt="system",
            user_input="user",
            handler=BaseHandler(),
            tools_schema=[],
            max_turns=2,
            on_event=events.append,
        )

        chunks = [
            event["content"]["text"]
            for event in events
            if event.get("sessionUpdate") == "agent_message_chunk"
        ]
        self.assertEqual(chunks, ["诊断结论: scene home_feed19 is not configured"])
        self.assertIn("最终报告正文", client.new_messages[1][0]["content"])
        self.assertEqual(exit_reason["result"], "NO_TOOL_CALL")

    def test_internal_thinking_with_summary_retries_for_visible_final_answer(self):
        client = SequenceFakeClient([
            (
                "<clinical_thinking>Need to enter Step 6 and write the report.</clinical_thinking>\n"
                "<summary>confirmed root cause</summary>"
            ),
            "诊断结论: scene home_feed19 is not configured",
        ])
        events = []

        exit_reason = agent_runner_loop(
            client=client,
            system_prompt="system",
            user_input="user",
            handler=BaseHandler(),
            tools_schema=[],
            max_turns=2,
            on_event=events.append,
        )

        chunks = [
            event["content"]["text"]
            for event in events
            if event.get("sessionUpdate") == "agent_message_chunk"
        ]
        self.assertEqual(chunks, ["诊断结论: scene home_feed19 is not configured"])
        self.assertIn("内部思考/规划标签", client.new_messages[1][0]["content"])
        self.assertEqual(exit_reason["result"], "NO_TOOL_CALL")

    def test_internal_thinking_only_no_tool_call_emits_visible_fallback(self):
        events, exit_reason = self.run_loop_events("<clinical_thinking>service is running</clinical_thinking>")

        chunks = [
            event["content"]["text"]
            for event in events
            if event.get("sessionUpdate") == "agent_message_chunk"
        ]
        self.assertEqual(chunks, ["service is running"])
        self.assertEqual(exit_reason["result"], "NO_TOOL_CALL")

    def test_summary_only_emits_summary_when_retry_unavailable(self):
        events, exit_reason = self.run_loop_events("<summary>confirmed root cause</summary>")

        chunks = [
            event["content"]["text"]
            for event in events
            if event.get("sessionUpdate") == "agent_message_chunk"
        ]
        self.assertEqual(chunks, ["confirmed root cause"])
        self.assertEqual(exit_reason["result"], "NO_TOOL_CALL")

    def test_large_tool_result_is_persisted_before_next_turn(self):
        with tempfile.TemporaryDirectory() as output_root:
            client = ToolThenFinalClient([
                ToolCall(id="call-1", name="big_tool", input={}),
            ])
            events = []

            exit_reason = agent_runner_loop(
                client=client,
                system_prompt="system",
                user_input="user",
                handler=BigToolHandler(output_root),
                tools_schema=[],
                max_turns=2,
                on_event=events.append,
            )

            tool_message = client.new_messages[1][0]
            self.assertEqual(tool_message["role"], "tool")
            self.assertIn("<persisted-output>", tool_message["content"])
            self.assertIn("path:", tool_message["content"])
            self.assertLess(len(tool_message["content"]), 20000)

            saved_path = os.path.join(output_root, "tool_results", "tool-1-0.txt")
            with open(saved_path, encoding="utf-8") as file:
                self.assertEqual(file.read(), "x" * 120000)
            completed_update = [
                event for event in events
                if event.get("sessionUpdate") == "tool_call_update" and event.get("status") == "completed"
            ][0]
            self.assertTrue(completed_update["data"]["result_persisted"])
            self.assertLess(len(str(completed_update["data"])), 20000)
            self.assertEqual(exit_reason["result"], "NO_TOOL_CALL")


if __name__ == "__main__":
    unittest.main()
