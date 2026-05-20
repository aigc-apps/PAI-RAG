import asyncio
import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from session_store import SERVER_USER_ID, SessionStore
import backend.server as server


class SessionStoreResponseTests(unittest.TestCase):
    def test_save_load_latest_and_delete_response(self):
        with tempfile.TemporaryDirectory() as root:
            store = SessionStore(os.path.join(root, "sessions"))
            response = {"id": "resp_1", "object": "response", "output": []}
            history = [{"role": "user", "content": "hello"}]

            store.save_response(
                "resp_1",
                response,
                conversation_history=history,
                instructions="be brief",
                session_id="session_1",
                user_id=SERVER_USER_ID,
                conversation="conv_1",
            )

            loaded = store.load_response("resp_1", user_id=SERVER_USER_ID)
            self.assertEqual(loaded["response"], response)
            self.assertEqual(loaded["conversation_history"], history)
            self.assertEqual(loaded["instructions"], "be brief")
            self.assertEqual(store.latest_response_for_conversation("conv_1"), "resp_1")
            self.assertTrue(store.delete_response("resp_1", user_id=SERVER_USER_ID))
            self.assertIsNone(store.load_response("resp_1", user_id=SERVER_USER_ID))


class ResponseThinkingTagTests(unittest.TestCase):
    def test_taking_tag_is_stripped_from_visible_assistant_text(self):
        text = "hello\n<taking>private plan</taking>\nworld"
        self.assertEqual(server._strip_thinking_blocks(text).strip(), "hello\n\nworld")

    def test_working_tag_is_stripped_from_visible_assistant_text(self):
        text = "hello\n<working>private work</working>\nworld"
        self.assertEqual(server._strip_thinking_blocks(text).strip(), "hello\n\nworld")

    def test_skill_context_tag_is_stripped_from_visible_assistant_text(self):
        text = "hello\n<skill_context>private skill context</skill_context>\nworld"
        self.assertEqual(server._strip_thinking_blocks(text).strip(), "hello\n\nworld")

    def test_taking_tag_is_reconstructed_as_agent_step_event(self):
        updates = server._agent_updates_from_output([
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "<taking>private plan</taking>\nvisible answer",
                    },
                ],
            },
        ])

        self.assertEqual(updates[0]["sessionUpdate"], "thought_start")
        self.assertEqual(updates[1]["sessionUpdate"], "thought_delta")
        self.assertEqual(updates[1]["content"]["text"], "private plan")
        self.assertEqual(updates[2]["sessionUpdate"], "thought_done")
        self.assertEqual(updates[3]["sessionUpdate"], "agent_message_chunk")
        self.assertEqual(updates[3]["content"]["text"].strip(), "visible answer")

    def test_working_tag_is_reconstructed_as_agent_step_event(self):
        updates = server._agent_updates_from_output([
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "<working>private work</working>\nvisible answer",
                    },
                ],
            },
        ])

        self.assertEqual(updates[0]["sessionUpdate"], "thought_start")
        self.assertEqual(updates[1]["sessionUpdate"], "thought_delta")
        self.assertEqual(updates[1]["content"]["text"], "private work")
        self.assertEqual(updates[2]["sessionUpdate"], "thought_done")
        self.assertEqual(updates[3]["sessionUpdate"], "agent_message_chunk")
        self.assertEqual(updates[3]["content"]["text"].strip(), "visible answer")

    def test_skill_context_tag_is_reconstructed_as_agent_step_event(self):
        updates = server._agent_updates_from_output([
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "<skill_context>private skill context</skill_context>\nvisible answer",
                    },
                ],
            },
        ])

        self.assertEqual(updates[0]["sessionUpdate"], "thought_start")
        self.assertEqual(updates[1]["sessionUpdate"], "thought_delta")
        self.assertEqual(updates[1]["content"]["text"], "private skill context")
        self.assertEqual(updates[2]["sessionUpdate"], "thought_done")
        self.assertEqual(updates[3]["sessionUpdate"], "agent_message_chunk")
        self.assertEqual(updates[3]["content"]["text"].strip(), "visible answer")

    def test_final_report_contract_text_is_authoritative(self):
        text = server._final_assistant_text([
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "<summary>done</summary>"}],
            },
            {
                "type": "message",
                "metadata": {"pairag": {"is_final_report": True}},
                "content": [{"type": "output_text", "text": "## Report\nReadable result."}],
            },
        ])

        self.assertEqual(text, "## Report\nReadable result.")

    def test_process_reasoning_message_reconstructs_thought_not_answer(self):
        updates = server._agent_updates_from_output([
            {
                "type": "message",
                "metadata": {"pairag": {"is_process_reasoning": True}},
                "content": [{"type": "output_text", "text": "Let me check.\n<taking>use tool</taking>"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "code_run",
                "arguments": "{}",
            },
        ])

        self.assertEqual(updates[0]["sessionUpdate"], "thought_start")
        thought_text = "".join(
            update["content"]["text"]
            for update in updates
            if update["sessionUpdate"] == "thought_delta"
        )
        self.assertIn("Let me check.", thought_text)
        self.assertIn("use tool", thought_text)
        self.assertFalse(any(update["sessionUpdate"] == "agent_message_chunk" for update in updates))

    def test_reasoning_output_item_reconstructs_thought_not_answer(self):
        updates = server._agent_updates_from_output([
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "Let me check."}],
                "metadata": {"pairag": {"is_process_reasoning": True}},
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Final answer."}],
            },
        ])

        self.assertEqual(updates[0]["sessionUpdate"], "thought_start")
        self.assertEqual(updates[1]["content"]["text"], "Let me check.")
        self.assertTrue(any(
            update["sessionUpdate"] == "agent_message_chunk"
            and update["content"]["text"] == "Final answer."
            for update in updates
        ))

    def test_process_reasoning_message_is_not_final_assistant_text(self):
        text = server._final_assistant_text([
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "Reasoning."}],
            },
            {
                "type": "message",
                "metadata": {"pairag": {"is_process_reasoning": True}},
                "content": [{"type": "output_text", "text": "Let me check."}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Final answer."}],
            },
        ])

        self.assertEqual(text, "Final answer.")


class FinalizeOutputForResponsesTests(unittest.TestCase):
    """``_finalize_output_for_responses`` enforces the single-`message` invariant
    so OpenAI SDK consumers' ``response.output_text`` accessor naturally yields
    the final answer.
    """

    def _state(self, output):
        return SimpleNamespace(output=list(output))

    def test_lone_plain_message_no_tools_gets_flagged_as_final(self):
        from backend.agents_sdk.runner import _finalize_output_for_responses

        state = self._state([
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "你好"}],
            },
        ])
        _finalize_output_for_responses(state)

        self.assertEqual(len(state.output), 1)
        self.assertEqual(state.output[0]["type"], "message")
        self.assertTrue(state.output[0]["metadata"]["pairag"]["is_final_report"])

    def test_final_report_present_other_message_becomes_reasoning(self):
        from backend.agents_sdk.runner import _finalize_output_for_responses

        state = self._state([
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "intermediate prose"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "code_run",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            },
            {
                "type": "message",
                "role": "assistant",
                "metadata": {"pairag": {"is_final_report": True}},
                "content": [{"type": "output_text", "text": "## Final report"}],
            },
        ])
        _finalize_output_for_responses(state)

        message_indices = [i for i, item in enumerate(state.output) if item.get("type") == "message"]
        self.assertEqual(len(message_indices), 1)
        self.assertEqual(state.output[message_indices[0]]["content"][0]["text"], "## Final report")

        intermediate = state.output[0]
        self.assertEqual(intermediate["type"], "reasoning")
        self.assertTrue(intermediate["metadata"]["pairag"]["is_process_reasoning"])
        self.assertEqual(intermediate["content"][0]["type"], "summary_text")
        self.assertEqual(intermediate["content"][0]["text"], "intermediate prose")

    def test_tool_calls_no_final_report_leaves_plain_message_untouched(self):
        """When tools were used but the model didn't call ``final_report``, the
        upstream ``_needs_final_report_retry`` is responsible for fixing it.
        ``_finalize_output_for_responses`` must not auto-flag a plain message in
        that case — that would short-circuit the retry contract.
        """
        from backend.agents_sdk.runner import _finalize_output_for_responses

        state = self._state([
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "code_run",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "post-tool prose"}],
            },
        ])
        _finalize_output_for_responses(state)

        message = state.output[-1]
        self.assertEqual(message["type"], "message")
        self.assertNotIn("metadata", message)


class BackgroundReviewSchedulingTests(unittest.TestCase):
    def _session(self, memory_root):
        class _Session:
            sid = "sess_1"
            user_id = SERVER_USER_ID
            memory_scope = SimpleNamespace(root=memory_root)
            ui_msgs = []
            _lock = threading.RLock()

            def save(self):
                pass

        return _Session()

    def test_completed_turn_schedules_background_review_after_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = self._session(tmp)
            archive_path = os.path.join(tmp, "L4_raw_sessions", "session.md")

            with mock.patch.object(server, "_archive_session_for_replay", return_value=archive_path), \
                 mock.patch.object(server, "schedule_background_memory_review", return_value=True) as schedule:
                result = server._append_turn_to_session(
                    sess,
                    user_text="请诊断",
                    assistant_text="诊断完成",
                    final_status="completed",
                    run_id="resp_1",
                    active_skill="diagnosis",
                )

        self.assertEqual(result, archive_path)
        schedule.assert_called_once()
        kwargs = schedule.call_args.kwargs
        self.assertEqual(kwargs["session_id"], "sess_1")
        self.assertEqual(kwargs["run_id"], "resp_1")
        self.assertEqual(kwargs["memory_root"], tmp)
        self.assertEqual(kwargs["archive_path"], archive_path)
        self.assertTrue(kwargs["long_term_enabled"])
        self.assertEqual(kwargs["active_skill"], "diagnosis")
        self.assertEqual(kwargs["llm_history"], [
            {"role": "user", "content": "请诊断"},
            {"role": "assistant", "content": "诊断完成"},
        ])

    def test_non_completed_turn_does_not_schedule_background_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = self._session(tmp)
            archive_path = os.path.join(tmp, "L4_raw_sessions", "session.md")

            with mock.patch.object(server, "_archive_session_for_replay", return_value=archive_path), \
                 mock.patch.object(server, "schedule_background_memory_review") as schedule:
                server._append_turn_to_session(
                    sess,
                    user_text="继续",
                    assistant_text="",
                    final_status="requires_action",
                    run_id="resp_pause",
                )

        schedule.assert_not_called()

    def test_empty_archive_path_does_not_schedule_background_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = self._session(tmp)

            with mock.patch.object(server, "_archive_session_for_replay", return_value=""), \
                 mock.patch.object(server, "schedule_background_memory_review") as schedule:
                server._append_turn_to_session(
                    sess,
                    user_text="请诊断",
                    assistant_text="诊断完成",
                    final_status="completed",
                    run_id="resp_1",
                )

        schedule.assert_not_called()

    def test_background_review_scheduler_failure_does_not_break_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = self._session(tmp)
            archive_path = os.path.join(tmp, "L4_raw_sessions", "session.md")

            with mock.patch.object(server, "_archive_session_for_replay", return_value=archive_path), \
                 mock.patch.object(server, "schedule_background_memory_review", side_effect=RuntimeError("boom")):
                result = server._append_turn_to_session(
                    sess,
                    user_text="请诊断",
                    assistant_text="诊断完成",
                    final_status="completed",
                    run_id="resp_1",
                )

        self.assertEqual(result, archive_path)
        self.assertEqual(len(sess.ui_msgs), 2)


class CollectResponsesCompletionTests(unittest.TestCase):
    """``_collect_responses_completion`` drains the SSE generator the
    streaming path uses, and returns the terminal response object as a
    plain dict (FastAPI auto-serializes). Pre-flight error events become
    HTTP 4xx so non-stream consumers see proper status codes.
    """

    def _sse(self, event_type, payload):
        body = dict(payload)
        body.setdefault('type', event_type)
        return f'event: {event_type}\ndata: {json.dumps(body)}\n\n'

    async def _drain(self, chunks):
        async def gen():
            for chunk in chunks:
                yield chunk

        return await server._collect_responses_completion(gen())

    def test_completed_event_returns_response_object_without_sse_keys(self):
        completed = {
            'id': 'resp_1',
            'object': 'response',
            'status': 'completed',
            'model': 'qwen-plus',
            'output': [{'type': 'message', 'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': 'ok'}]}],
            'output_text': 'ok',
            'usage': {'total_tokens': 5},
        }
        chunks = [
            self._sse('response.created', {'id': 'resp_1', 'status': 'in_progress'}),
            self._sse('response.output_text.delta', {'delta': 'ok'}),
            self._sse('response.completed', completed),
            'data: [DONE]\n\n',
        ]
        result = asyncio.run(self._drain(chunks))
        self.assertNotIsInstance(result, server.JSONResponse)
        self.assertEqual(result['id'], 'resp_1')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['output_text'], 'ok')
        self.assertNotIn('type', result)
        self.assertNotIn('sequence_number', result)

    def test_preflight_response_not_found_returns_404(self):
        chunks = [
            self._sse('response.failed', {
                'response_id': 'resp_missing',
                'error': {'message': 'Response not found: resp_missing',
                          'code': 'response_not_found'},
            }),
            'data: [DONE]\n\n',
        ]
        result = asyncio.run(self._drain(chunks))
        self.assertIsInstance(result, server.JSONResponse)
        self.assertEqual(result.status_code, 404)
        body = json.loads(result.body)
        self.assertEqual(body['error']['code'], 'response_not_found')

    def test_preflight_invalid_input_returns_400(self):
        chunks = [
            self._sse('response.failed', {
                'response_id': '',
                'error': {'message': 'No user message in input',
                          'code': 'invalid_input'},
            }),
            'data: [DONE]\n\n',
        ]
        result = asyncio.run(self._drain(chunks))
        self.assertIsInstance(result, server.JSONResponse)
        self.assertEqual(result.status_code, 400)

    def test_terminal_failed_run_returns_200_with_response_body(self):
        # Run-level failure: payload has object="response" so it's NOT
        # the pre-flight branch — return 200 with the response object so
        # the client can inspect status="failed" + error in the body,
        # mirroring OpenAI's behavior.
        failed = {
            'id': 'resp_2',
            'object': 'response',
            'status': 'failed',
            'model': 'qwen-plus',
            'output': [],
            'error': {'message': 'tool blew up'},
        }
        chunks = [
            self._sse('response.failed', failed),
            'data: [DONE]\n\n',
        ]
        result = asyncio.run(self._drain(chunks))
        self.assertNotIsInstance(result, server.JSONResponse)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error']['message'], 'tool blew up')

    def test_requires_action_returns_response_body(self):
        ra = {
            'id': 'resp_3',
            'object': 'response',
            'status': 'requires_action',
            'model': 'qwen-plus',
            'output': [],
            'required_action': {'type': 'submit_tool_outputs',
                                'submit_tool_outputs': {'tool_calls': []}},
        }
        chunks = [
            self._sse('response.requires_action', ra),
            self._sse('response.incomplete', dict(ra, status='incomplete')),
            'data: [DONE]\n\n',
        ]
        result = asyncio.run(self._drain(chunks))
        self.assertNotIsInstance(result, server.JSONResponse)
        # ``response.requires_action`` arrives first; the incomplete duplicate
        # for SDK strict-mode parity must not overwrite it.
        self.assertEqual(result['status'], 'requires_action')

    def test_no_terminal_event_returns_500(self):
        # Defensive: if the generator finishes without ever emitting a
        # terminal event, that's a server bug — surface as 500.
        chunks = [
            self._sse('response.created', {'id': 'resp_x'}),
            'data: [DONE]\n\n',
        ]
        result = asyncio.run(self._drain(chunks))
        self.assertIsInstance(result, server.JSONResponse)
        self.assertEqual(result.status_code, 500)


if __name__ == "__main__":
    unittest.main()
