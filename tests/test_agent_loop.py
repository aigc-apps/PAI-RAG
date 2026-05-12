import os
import tempfile
import unittest

from agent_loop import BaseHandler, StepOutcome, agent_runner_loop, sanitize_for_archive
from llm_client import Response, ToolCall


class FakeClient:
    def __init__(self, content):
        self.content = content

    def chat(self, system, new_messages, tools):
        yield self.content
        return Response(content=self.content)


class HistoryFakeClient:
    def __init__(self, content, chunk_size=None):
        self.content = content
        self.chunk_size = chunk_size or len(content)
        self.history = []

    def chat(self, system, new_messages, tools):
        self.history.extend(new_messages)
        for offset in range(0, len(self.content), self.chunk_size):
            yield self.content[offset:offset + self.chunk_size]
        self.history.append({"role": "assistant", "content": self.content})
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


class BigFileReadHandler(BigToolHandler):
    def do_file_read(self, args, response):
        return StepOutcome("r" * 15000, next_prompt="continue")


class SimpleToolHandler(BaseHandler):
    def do_simple_tool(self, args, response):
        return StepOutcome({"status": "success", "value": "checked"}, next_prompt="continue")


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

    def test_text_mode_no_tool_call_streams_final_answer_once(self):
        chunks = []

        exit_reason = agent_runner_loop(
            client=FakeClient("final answer"),
            system_prompt="system",
            user_input="user",
            handler=BaseHandler(),
            tools_schema=[],
            max_turns=1,
            on_chunk=chunks.append,
        )

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

    def test_text_mode_summary_only_retry_falls_back_to_summary(self):
        client = SequenceFakeClient([
            "<summary>查询当前系统日期和星期</summary>",
            "<summary>获取当前系统日期和星期</summary>",
        ])
        chunks = []

        exit_reason = agent_runner_loop(
            client=client,
            system_prompt="system",
            user_input="user",
            handler=BaseHandler(),
            tools_schema=[],
            max_turns=2,
            on_chunk=chunks.append,
        )

        self.assertEqual(chunks, ["获取当前系统日期和星期"])
        self.assertIn("不要只输出 <summary>", client.new_messages[1][0]["content"])
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

    def test_large_assistant_output_is_persisted_and_stream_limited(self):
        content = "<file_content>" + ("x" * 30000) + "</file_content>"
        with tempfile.TemporaryDirectory() as output_root:
            client = HistoryFakeClient(content, chunk_size=5000)
            events = []

            exit_reason = agent_runner_loop(
                client=client,
                system_prompt="system",
                user_input="user",
                handler=BigToolHandler(output_root),
                tools_schema=[],
                max_turns=1,
                on_event=events.append,
            )

            saved_path = os.path.join(output_root, "assistant_outputs", "model-1.txt")
            with open(saved_path, encoding="utf-8") as file:
                self.assertEqual(file.read(), content)

            self.assertEqual(exit_reason["result"], "NO_TOOL_CALL")
            self.assertIn("<persisted-output>", exit_reason["data"])
            self.assertIn("assistant file_content output too large", exit_reason["data"])
            self.assertLess(len(exit_reason["data"]), 14000)
            self.assertIn("<persisted-output>", client.history[-1]["content"])
            self.assertLess(len(client.history[-1]["content"]), 14000)

            thought_text = "".join(
                event.get("content", {}).get("text", "")
                for event in events
                if event.get("sessionUpdate") == "thought_delta"
            )
            self.assertIn("ASSISTANT OUTPUT STREAM TRUNCATED", thought_text)
            self.assertLess(len(thought_text), 13000)

            chunks = [
                event["content"]["text"]
                for event in events
                if event.get("sessionUpdate") == "agent_message_chunk"
            ]
            self.assertEqual(len(chunks), 1)
            self.assertIn("<persisted-output>", chunks[0])
            self.assertLess(len(chunks[0]), 14000)

    def test_large_file_read_result_is_persisted_before_next_turn(self):
        with tempfile.TemporaryDirectory() as output_root:
            client = ToolThenFinalClient([
                ToolCall(id="call-1", name="file_read", input={"path": "large.txt"}),
            ])
            events = []

            exit_reason = agent_runner_loop(
                client=client,
                system_prompt="system",
                user_input="user",
                handler=BigFileReadHandler(output_root),
                tools_schema=[],
                max_turns=2,
                on_event=events.append,
            )

            tool_message = client.new_messages[1][0]
            self.assertEqual(tool_message["role"], "tool")
            self.assertIn("<persisted-output>", tool_message["content"])
            self.assertIn("path:", tool_message["content"])
            self.assertLess(len(tool_message["content"]), 14000)

            saved_path = os.path.join(output_root, "tool_results", "tool-1-0.txt")
            with open(saved_path, encoding="utf-8") as file:
                self.assertEqual(file.read(), "r" * 15000)
            self.assertEqual(exit_reason["result"], "NO_TOOL_CALL")

    def test_archive_sanitization_redacts_secrets_and_caps_long_text(self):
        payload = {
            "result": "NO_TOOL_CALL",
            "data": (
                '"FeatureDBPassword": "supersecretvalue1234567890"\n'
                "api_key = sk-test1234567890abcdefghijklmnopqrstuvwxyz\n"
                + ("x" * 40000)
            ),
        }

        sanitized = sanitize_for_archive(payload, max_text_chars=1000)
        text = sanitized["data"]

        self.assertIn("[REDACTED]", text)
        self.assertNotIn("supersecretvalue1234567890", text)
        self.assertNotIn("sk-test1234567890abcdefghijklmnopqrstuvwxyz", text)
        self.assertIn("OUTPUT PREVIEW TRUNCATED", text)
        self.assertLess(len(text), 1300)

    def test_max_turns_exceeded_emits_fallback_report(self):
        client = ToolThenFinalClient(
            [ToolCall(id="call-1", name="simple_tool", input={})],
            final_content="部分完成报告: 已完成检查，仍需下一步处理",
        )
        events = []

        exit_reason = agent_runner_loop(
            client=client,
            system_prompt="system",
            user_input="user",
            handler=SimpleToolHandler(),
            tools_schema=[],
            max_turns=1,
            on_event=events.append,
        )

        chunks = [
            event["content"]["text"]
            for event in events
            if event.get("sessionUpdate") == "agent_message_chunk"
        ]
        self.assertEqual(chunks, ["部分完成报告: 已完成检查，仍需下一步处理"])
        self.assertEqual(exit_reason["result"], "MAX_TURNS_EXCEEDED")
        self.assertIn("部分完成报告", exit_reason["data"])
        self.assertEqual(
            client.new_messages[1][-1]["content"].splitlines()[0],
            "已达到本次 agent 最大执行轮次，不能再调用工具。",
        )


if __name__ == "__main__":
    unittest.main()
