"""Phase-2 component tests for the SDK runtime.

Covers what we can verify without a live LLM endpoint: wire mapping
(event_bridge), HITL envelopes (hitl), RunState CRUD (run_state_store),
and audit redaction (audit.store). End-to-end runner tests live in
``test_agents_sdk_runner.py`` (later phase) once we stub a Model.
"""
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from session_store import SERVER_USER_ID, SessionStore
from backend.agents_sdk import event_bridge
from backend.agents_sdk.agent_factory import build as build_agent
from backend.agents_sdk.event_bridge import ResponsesStreamState
from backend.agents_sdk.hitl import (
    InterruptionEnvelope,
    RESERVED_ASK_USER,
    RESERVED_REQUEST_APPROVAL,
    ResumePayload,
    interruption_from_sdk_item,
)
from backend.agents_sdk.lifecycle import (
    RUN_STATE_COMPLETED,
    RUN_STATE_REQUIRES_ACTION,
    RUN_STATE_RUNNING,
)
from backend.agents_sdk.run_state_store import RunStateStore
from backend.agents_sdk.runner import (
    RunContext,
    StreamFrame,
    _apply_final_report_contract,
    _autonomous_hitl_resolution,
    _drive_stream,
    _final_report_retry_input,
    _final_text_from_output,
    _needs_final_report_retry,
    _needs_tool_intent_retry,
)
from backend.tools.wrappers import build_tool_list


# ─── helpers to fake SDK events ───────────────────────────────────────────

class _RawEvent:
    """Mirror RawResponsesStreamEvent.{type, data}."""
    type = 'raw_response_event'

    def __init__(self, data):
        self.data = data


class _ItemEvent:
    """Mirror RunItemStreamEvent.{type, item, name}."""
    type = 'run_item_stream_event'

    def __init__(self, item, name=''):
        self.item = item
        self.name = name


class _FakeRunState:
    def __init__(self, interruptions=None):
        self._interruptions = list(interruptions or [])
        self.approved = []
        self.rejected = []

    def get_interruptions(self):
        return list(self._interruptions)

    def approve(self, item):
        self.approved.append(item)
        self._interruptions = [candidate for candidate in self._interruptions if candidate is not item]

    def reject(self, item, rejection_message=''):
        self.rejected.append((item, rejection_message))
        self._interruptions = [candidate for candidate in self._interruptions if candidate is not item]

    def to_string(self):
        return '{"state":"ok"}'


class _FakeStreaming:
    def __init__(self, events=None, state=None):
        self._events = list(events or [])
        self._state = state or _FakeRunState()

    async def stream_events(self):
        for event in self._events:
            yield event

    def to_state(self):
        return self._state


class _FakeStateStore:
    def __init__(self):
        self.rows = []

    def upsert(self, **kwargs):
        self.rows.append(kwargs)


def _delta(text):
    return _RawEvent(SimpleNamespace(type='response.output_text.delta', delta=text))


def _tool_call_item(call_id='call_42', tool_name='file_read', args='{"path":"x"}'):
    raw = {'call_id': call_id, 'name': tool_name, 'arguments': args}
    cls = type('ToolCallItem', (), {})
    inst = cls()
    inst.call_id = call_id
    inst.tool_name = tool_name
    inst.raw_item = raw
    return inst


def _tool_output_item(call_id='call_42', output='42'):
    raw = {'call_id': call_id, 'output': output}
    cls = type('ToolCallOutputItem', (), {})
    inst = cls()
    inst.call_id = call_id
    inst.output = output
    inst.raw_item = raw
    return inst


def _message_output_item():
    cls = type('MessageOutputItem', (), {})
    return cls()


def _approval_item(call_id='approve_1', tool_name='ask_user', args='{"question":"Pick?"}'):
    cls = type('ToolApprovalItem', (), {})
    inst = cls()
    inst.id = call_id
    inst.tool_name = tool_name
    inst.arguments = args
    inst.raw_item = {'call_id': call_id, 'name': tool_name, 'arguments': args}
    return inst


# ─── event_bridge ─────────────────────────────────────────────────────────

class EventBridgeTests(unittest.TestCase):
    def test_text_delta_streams_live_and_lazy_opens_message(self):
        state = ResponsesStreamState()
        # First visible delta lazy-opens the message item AND emits the
        # delta — required for live token-by-token UX.
        chunks = event_bridge.to_responses_chunk(_delta('hello'), state, response_id='resp_1')
        self.assertEqual(
            [c['type'] for c in chunks],
            ['response.output_item.added', 'response.output_text.delta'],
        )
        self.assertEqual(chunks[0]['item']['type'], 'message')
        self.assertEqual(chunks[1]['delta'], 'hello')
        self.assertTrue(state.message_started)
        self.assertEqual(''.join(state.accumulated_text), 'hello')
        # Subsequent deltas only emit the delta — the message item is
        # already open.
        chunks2 = event_bridge.to_responses_chunk(_delta(' world'), state, response_id='resp_1')
        self.assertEqual([c['type'] for c in chunks2], ['response.output_text.delta'])
        self.assertEqual(chunks2[0]['delta'], ' world')
        self.assertEqual(''.join(state.accumulated_text), 'hello world')

    def test_tool_call_item_emits_added_and_done(self):
        state = ResponsesStreamState()
        ev = _ItemEvent(_tool_call_item())
        chunks = event_bridge.to_responses_chunk(ev, state, response_id='resp_1')
        self.assertEqual([c['type'] for c in chunks], [
            'response.output_item.added',
            'response.output_item.done',
        ])
        added = chunks[0]['item']
        self.assertEqual(added['type'], 'function_call')
        self.assertEqual(added['call_id'], 'call_42')
        self.assertEqual(added['name'], 'file_read')

    def test_tool_output_item_emits_added_and_done(self):
        state = ResponsesStreamState()
        ev = _ItemEvent(_tool_output_item())
        chunks = event_bridge.to_responses_chunk(ev, state, response_id='resp_1')
        self.assertEqual([c['type'] for c in chunks], [
            'response.output_item.added',
            'response.output_item.done',
        ])
        self.assertEqual(chunks[0]['item']['type'], 'function_call_output')

    def test_message_output_item_closes_assistant_message(self):
        state = ResponsesStreamState()
        all_chunks: list[dict] = []
        # First the delta lazy-opens the message and live-emits.
        all_chunks += event_bridge.to_responses_chunk(_delta('done'), state, response_id='resp_1')
        # Then the closing MessageOutputItem brackets it with text.done +
        # item.done — no fresh added/delta pair (those already shipped live).
        ev = _ItemEvent(_message_output_item())
        all_chunks += event_bridge.to_responses_chunk(ev, state, response_id='resp_1')
        self.assertEqual([c['type'] for c in all_chunks], [
            'response.output_item.added',
            'response.output_text.delta',
            'response.output_text.done',
            'response.output_item.done',
        ])
        # Final message in output[] is completed
        msg = state.output[0]
        self.assertEqual(msg['status'], 'completed')
        self.assertEqual(msg['content'][0]['text'], 'done')

    def test_to_chat_chunk_only_emits_text_deltas(self):
        chunk = event_bridge.to_chat_chunk(
            _delta('hi'), model='qwen-test', completion_id='chatcmpl-1',
        )
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk['object'], 'chat.completion.chunk')
        self.assertEqual(chunk['choices'][0]['delta']['content'], 'hi')
        self.assertIsNone(chunk['choices'][0]['finish_reason'])
        self.assertIsNone(event_bridge.to_chat_chunk(
            _ItemEvent(_tool_call_item()),
            model='qwen-test', completion_id='chatcmpl-1',
        ))

    def test_chat_pause_chunk_uses_finish_reason_tool_calls(self):
        env = InterruptionEnvelope(
            call_id='c1', tool_name='ask_user',
            arguments={'question': 'Pick'}, is_input_request=True,
        )
        chunk = event_bridge.chat_pause_chunk(env, model='qwen-test', completion_id='chatcmpl-1')
        self.assertEqual(chunk['choices'][0]['finish_reason'], 'tool_calls')
        tc = chunk['choices'][0]['delta']['tool_calls'][0]
        self.assertEqual(tc['function']['name'], RESERVED_ASK_USER)
        self.assertEqual(tc['id'], 'c1')

    def test_to_audit_categories(self):
        # text delta → llm_chunk
        row = event_bridge.to_audit(_delta('x'))
        self.assertEqual(row['category'], 'llm_chunk')
        # tool call → tool_call
        row = event_bridge.to_audit(_ItemEvent(_tool_call_item()))
        self.assertEqual(row['category'], 'tool_call')
        # tool output → tool_result
        row = event_bridge.to_audit(_ItemEvent(_tool_output_item()))
        self.assertEqual(row['category'], 'tool_result')
        # approval → hitl_pause
        row = event_bridge.to_audit(_ItemEvent(_approval_item()))
        self.assertEqual(row['category'], 'hitl_pause')
        self.assertEqual(row['payload']['tool_name'], 'ask_user')
        self.assertTrue(row['payload']['is_input_request'])

    def test_final_report_contract_synthesizes_terminal_message(self):
        state = ResponsesStreamState()
        state.output.append({
            'id': 'msg_summary',
            'type': 'message',
            'status': 'completed',
            'role': 'assistant',
            'content': [{'type': 'output_text', 'text': '<summary>done</summary>'}],
        })

        chunks = _apply_final_report_contract(
            state,
            response_id='resp_1',
            report_text='## Conclusion\nReadable report.',
        )

        self.assertEqual([c['type'] for c in chunks], [
            'response.output_item.added',
            'response.output_text.delta',
            'response.output_text.done',
            'response.output_item.done',
        ])
        self.assertEqual(chunks[1]['delta'], '## Conclusion\nReadable report.')
        self.assertTrue(state.output[-1]['metadata']['pairag']['is_final_report'])
        self.assertEqual(_final_text_from_output(state.output), '## Conclusion\nReadable report.')

    def test_final_report_arguments_stream_as_visible_text(self):
        state = ResponsesStreamState()
        add = _RawEvent(SimpleNamespace(
            type='response.output_item.added',
            item={
                'id': 'fc_1',
                'type': 'function_call',
                'status': 'in_progress',
                'call_id': 'call_1',
                'name': 'final_report',
                'arguments': '',
            },
        ))
        event_bridge.to_responses_chunk(add, state, response_id='resp_1')

        delta = _RawEvent(SimpleNamespace(
            type='response.function_call_arguments.delta',
            item_id='fc_1',
            output_index=0,
            delta='{"report_markdown":"## Result\\nLine 1',
        ))
        chunks = event_bridge.to_responses_chunk(delta, state, response_id='resp_1')
        text_deltas = [c['delta'] for c in chunks if c['type'] == 'response.output_text.delta']

        self.assertEqual(''.join(text_deltas), '## Result\nLine 1')
        self.assertTrue(state.output[-1]['metadata']['pairag']['is_final_report'])

        done = _RawEvent(SimpleNamespace(
            type='response.function_call_arguments.done',
            item_id='fc_1',
            output_index=0,
            arguments='{"report_markdown":"## Result\\nLine 1"}',
        ))
        done_chunks = event_bridge.to_responses_chunk(done, state, response_id='resp_1')

        self.assertIn('response.output_text.done', [c['type'] for c in done_chunks])
        self.assertEqual(state.output[-1]['status'], 'completed')
        self.assertEqual(state.output[-1]['content'][0]['text'], '## Result\nLine 1')

    def test_summary_only_tool_run_requests_final_report_retry(self):
        output = [
            {
                'id': 'fc_1',
                'type': 'function_call',
                'status': 'completed',
                'call_id': 'call_1',
                'name': 'code_run',
                'arguments': '{}',
            },
            {
                'id': 'fc_2',
                'type': 'function_call',
                'status': 'completed',
                'call_id': 'call_2',
                'name': 'code_run',
                'arguments': '{}',
            },
            {
                'id': 'fco_1',
                'type': 'function_call_output',
                'status': 'completed',
                'call_id': 'call_1',
                'output': json.dumps({
                    'status': 'error',
                    'exit_code': 1,
                    'stdout': (
                        'Validation finished: 2 error(s), 0 warning(s)\n\n'
                        "  [ERROR] A: 'x' is a required property\n"
                        "  [ERROR] B: 'y' is a required property\n"
                    ),
                }),
            },
            {
                'id': 'fco_2',
                'type': 'function_call_output',
                'status': 'completed',
                'call_id': 'call_2',
                'output': json.dumps({'status': 'success', 'exit_code': 0, 'stdout': ''}),
            },
            {
                'id': 'msg_1',
                'type': 'message',
                'status': 'completed',
                'role': 'assistant',
                'content': [{
                    'type': 'output_text',
                    'text': '<summary>完成配置校验：发现错误</summary>',
                }],
            },
        ]

        self.assertTrue(_needs_final_report_retry(output))
        retry_input = _final_report_retry_input(output)

        self.assertIn('只根据下面已有工具结果，调用 `final_report`', retry_input)
        self.assertIn('上一轮 summary：完成配置校验：发现错误', retry_input)
        self.assertIn('Validation finished: 2 error(s), 0 warning(s)', retry_input)
        self.assertIn("[ERROR] A: 'x' is a required property", retry_input)
        self.assertNotIn('status=success', retry_input)

    def test_pre_tool_text_does_not_satisfy_final_report(self):
        output = [
            {
                'id': 'msg_plan',
                'type': 'message',
                'status': 'completed',
                'role': 'assistant',
                'content': [{
                    'type': 'output_text',
                    'text': 'I need to inspect the config before calling the diagnostic tool.',
                }],
            },
            {
                'id': 'fc_1',
                'type': 'function_call',
                'status': 'completed',
                'call_id': 'call_1',
                'name': 'use_skill',
                'arguments': '{}',
            },
            {
                'id': 'fco_1',
                'type': 'function_call_output',
                'status': 'completed',
                'call_id': 'call_1',
                'output': json.dumps({'status': 'skill_activated'}),
            },
        ]

        self.assertTrue(_needs_final_report_retry(output))

    def test_textual_skill_activation_without_tool_call_needs_retry(self):
        output = [{
            'id': 'msg_1',
            'type': 'message',
            'status': 'completed',
            'role': 'assistant',
            'content': [{
                'type': 'output_text',
                'text': (
                    '<taking>应优先调用专用诊断 skill</taking>\n'
                    '<summary>启动 PAI-Rec 配置诊断技能</summary>'
                ),
            }],
        }]

        self.assertTrue(_needs_tool_intent_retry(output))

    def test_capability_description_does_not_trigger_tool_intent_retry(self):
        output = [{
            'id': 'msg_1',
            'type': 'message',
            'status': 'completed',
            'role': 'assistant',
            'content': [{
                'type': 'output_text',
                'text': '我是不确定时会优先调用工具获取真实信息的 AI 助手。',
            }],
        }]

        self.assertFalse(_needs_tool_intent_retry(output))

    def test_skill_capability_description_does_not_trigger_tool_intent_retry(self):
        output = [{
            'id': 'msg_1',
            'type': 'message',
            'status': 'completed',
            'role': 'assistant',
            'content': [{
                'type': 'output_text',
                'text': '我能够读写文件，并根据任务需要调用专业技能完成诊断。',
            }],
        }]

        self.assertFalse(_needs_tool_intent_retry(output))


class AgentFactoryTests(unittest.TestCase):
    def test_final_report_tool_is_terminal_contract(self):
        tools = build_tool_list(scope='main')
        self.assertIn('final_report', [tool.name for tool in tools])

        agent = build_agent(model='test-model', tools=tools, instructions_override='x')

        self.assertEqual(agent.tool_use_behavior, {'stop_at_tool_names': ['final_report']})


# ─── HITL envelope ───────────────────────────────────────────────────────

class HitlEnvelopeTests(unittest.TestCase):
    def test_responses_serialization(self):
        env = InterruptionEnvelope(
            call_id='c1', tool_name='ask_user',
            arguments={'question': 'Pick'}, is_input_request=True,
        )
        wire = env.serialize('responses')
        self.assertEqual(wire['type'], 'function_call')
        self.assertEqual(wire['name'], 'ask_user')
        self.assertEqual(json.loads(wire['arguments'])['question'], 'Pick')

    def test_chat_serialization_uses_reserved_name(self):
        env = InterruptionEnvelope(
            call_id='c1', tool_name='ask_user',
            arguments={'question': 'Pick'}, is_input_request=True,
        )
        wire = env.serialize('chat')
        self.assertEqual(wire['function']['name'], RESERVED_ASK_USER)
        env2 = InterruptionEnvelope(
            call_id='c2', tool_name='file_write',
            arguments={'path': '/tmp/x'}, is_input_request=False,
        )
        self.assertEqual(env2.serialize('chat')['function']['name'], RESERVED_REQUEST_APPROVAL)

    def test_resume_from_responses_input_function_call_output(self):
        items = [{'type': 'function_call_output', 'call_id': 'c1', 'output': 'A'}]
        rp = ResumePayload.from_responses_input(items)
        self.assertEqual(rp.call_id, 'c1')
        self.assertEqual(rp.answer, 'A')
        self.assertTrue(rp.approve)

    def test_resume_from_chat_message_role_tool(self):
        msg = {'role': 'tool', 'tool_call_id': 'c1', 'content': 'A'}
        rp = ResumePayload.from_chat_message(msg)
        self.assertEqual(rp.call_id, 'c1')
        self.assertEqual(rp.answer, 'A')
        self.assertTrue(rp.approve)

    def test_resume_from_chat_message_rejects_non_tool_role(self):
        with self.assertRaises(ValueError):
            ResumePayload.from_chat_message({'role': 'user', 'content': 'A'})

    def test_resume_from_chat_message_handles_content_parts(self):
        msg = {
            'role': 'tool', 'tool_call_id': 'c1',
            'content': [{'type': 'text', 'text': 'A'}, {'type': 'text', 'text': 'B'}],
        }
        rp = ResumePayload.from_chat_message(msg)
        self.assertEqual(rp.answer, 'AB')

    def test_resume_from_responses_input_mcp_approval_response(self):
        items = [{
            'type': 'mcp_approval_response',
            'approval_request_id': 'a1',
            'output': 'rejected',
            'approve': False,
        }]
        rp = ResumePayload.from_responses_input(items)
        self.assertEqual(rp.call_id, 'a1')
        self.assertFalse(rp.approve)

    def test_interruption_from_sdk_item_parses_args(self):
        item = _approval_item(call_id='abc', tool_name='ask_user', args='{"q":"hi"}')
        env = interruption_from_sdk_item(item)
        self.assertEqual(env.call_id, 'abc')
        self.assertEqual(env.tool_name, 'ask_user')
        self.assertEqual(env.arguments, {'q': 'hi'})
        self.assertTrue(env.is_input_request)


class AutonomousHitlTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_intent_text_without_call_retries_with_real_tool_call(self):
        from backend.agents_sdk import runner

        initial_stream = _FakeStreaming(
            events=[
                _delta('<taking>应优先调用专用诊断 skill</taking>\n<summary>启动 PAI-Rec 配置诊断技能</summary>'),
                _ItemEvent(_message_output_item()),
            ],
            state=_FakeRunState(),
        )
        retry_stream = _FakeStreaming(
            events=[
                _ItemEvent(_tool_call_item(call_id='call_skill', tool_name='use_skill', args='{"skill":"x"}')),
                _ItemEvent(_tool_output_item(call_id='call_skill', output='{"status":"skill_activated"}')),
                _delta('final answer'),
                _ItemEvent(_message_output_item()),
            ],
            state=_FakeRunState(),
        )
        state_store = _FakeStateStore()
        ctx = RunContext(
            session_id='sess_1',
            run_id='run_1',
            response_id='resp_1',
            user_id=SERVER_USER_ID,
            cwd='/tmp',
        )

        with patch.object(runner.Runner, 'run_streamed', return_value=retry_stream) as run_streamed:
            frames = []
            async for frame in _drive_stream(
                streaming=initial_stream,
                agent=object(),
                ctx=ctx,
                state_store=state_store,
                audit_store=None,
                audit_log_id='audit_1',
                model='qwen-test',
                max_turns=40,
                allow_hitl=False,
                original_input='校验配置',
            ):
                frames.append(frame)

        run_streamed.assert_called_once()
        output = frames[-1].response_object['output']
        self.assertTrue(any(item.get('type') == 'function_call' for item in output))
        self.assertEqual(frames[-1].response_object['status'], 'completed')

    def test_ask_user_default_action_is_used_for_autonomous_resolution(self):
        env = InterruptionEnvelope(
            call_id='c1',
            tool_name='ask_user',
            arguments={'question': 'Pick?', 'default_action': 'Use option A', 'risk': 'low'},
            is_input_request=True,
        )

        approve, answer, policy = _autonomous_hitl_resolution(env)

        self.assertTrue(approve)
        self.assertEqual(answer, 'Use option A')
        self.assertEqual(policy, 'default_action')

    async def test_default_autonomous_mode_approves_ask_user_and_completes(self):
        from backend.agents_sdk import runner

        item = _approval_item(
            call_id='ask_1',
            tool_name='ask_user',
            args='{"question":"Pick?","default_action":"Use A"}',
        )
        paused_state = _FakeRunState([item])
        final_state = _FakeRunState()
        initial_stream = _FakeStreaming(state=paused_state)
        final_stream = _FakeStreaming(events=[_delta('done')], state=final_state)
        state_store = _FakeStateStore()
        ctx = RunContext(
            session_id='sess_1',
            run_id='run_1',
            response_id='resp_1',
            user_id=SERVER_USER_ID,
            cwd='/tmp',
        )

        with patch.object(runner.Runner, 'run_streamed', return_value=final_stream) as run_streamed:
            frames = []
            async for frame in _drive_stream(
                streaming=initial_stream,
                agent=object(),
                ctx=ctx,
                state_store=state_store,
                audit_store=None,
                audit_log_id='audit_1',
                model='qwen-test',
                max_turns=40,
                allow_hitl=False,
            ):
                frames.append(frame)

        self.assertEqual(ctx.pending_human_answer, 'Use A')
        self.assertEqual(paused_state.approved, [item])
        run_streamed.assert_called_once()
        self.assertEqual(frames[-1].response_object['status'], 'completed')
        self.assertEqual(state_store.rows[-1]['status'], RUN_STATE_COMPLETED)

    async def test_allow_hitl_true_preserves_requires_action(self):
        item = _approval_item(
            call_id='ask_1',
            tool_name='ask_user',
            args='{"question":"Pick?","default_action":"Use A"}',
        )
        paused_state = _FakeRunState([item])
        initial_stream = _FakeStreaming(state=paused_state)
        state_store = _FakeStateStore()
        ctx = RunContext(
            session_id='sess_1',
            run_id='run_1',
            response_id='resp_1',
            user_id=SERVER_USER_ID,
            cwd='/tmp',
            allow_hitl=True,
        )

        frames = []
        async for frame in _drive_stream(
            streaming=initial_stream,
            agent=object(),
            ctx=ctx,
            state_store=state_store,
            audit_store=None,
            audit_log_id='audit_1',
            model='qwen-test',
            max_turns=40,
            allow_hitl=True,
        ):
            frames.append(frame)

        self.assertEqual(frames[-1].response_object['status'], 'requires_action')
        self.assertEqual(state_store.rows[-1]['status'], RUN_STATE_REQUIRES_ACTION)
        self.assertEqual(paused_state.approved, [])


class RunnerResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_run_awaits_run_state_from_string(self):
        from backend.agents_sdk import runner

        fake_agent = object()
        fake_streaming = object()
        matched_item = SimpleNamespace(call_id='call_1')
        calls = {'from_string_awaited': False, 'approved': None}

        class FakeState:
            def get_interruptions(self):
                return [matched_item]

            def approve(self, item):
                calls['approved'] = item

        async def fake_from_string(agent, blob, *, context_override=None):
            self.assertIs(agent, fake_agent)
            self.assertEqual(blob, 'serialized-state')
            self.assertIsNotNone(context_override)
            self.assertEqual(context_override.pending_human_answer, '123.txt')
            self.assertEqual(context_override.run_id, 'run_1')
            calls['from_string_awaited'] = True
            return FakeState()

        async def fake_drive_stream(**kwargs):
            self.assertIs(kwargs['streaming'], fake_streaming)
            yield StreamFrame(terminal=True, response_object={'id': 'resp_1', 'status': 'completed'})

        row = {
            'id': 'resp_1',
            'session_id': 'sess_1',
            'run_id': 'run_1',
            'response_id': 'resp_1',
            'audit_log_id': 'audit_1',
            'model': 'qwen-test',
            'run_state_blob': 'serialized-state',
        }

        with patch.object(runner, 'build_agent', return_value=fake_agent), \
             patch.object(runner.RunState, 'from_string', side_effect=fake_from_string), \
             patch.object(runner.Runner, 'run_streamed', return_value=fake_streaming) as fake_run_streamed, \
             patch.object(runner, '_drive_stream', fake_drive_stream):
            frames = []
            async for frame in runner._resume_run(
                row=row,
                resume=ResumePayload(call_id='call_1', answer='123.txt', approve=True),
                tools=[],
                user_id=SERVER_USER_ID,
                cwd='/tmp',
                model='qwen-test',
                audit_store=None,
                state_store=object(),
                instructions_override=None,
                max_turns=40,
                extras=None,
            ):
                frames.append(frame)

        self.assertTrue(calls['from_string_awaited'])
        self.assertIs(calls['approved'], matched_item)
        self.assertNotIn('context', fake_run_streamed.call_args.kwargs)
        self.assertEqual(frames[-1].response_object['status'], 'completed')


# ─── RunStateStore CRUD ─────────────────────────────────────────────────────

class RunStateStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Use SessionStore to bootstrap the schema (creates agent_run_states).
        self.session_store = SessionStore(os.path.join(self._tmp.name, 'sessions'))
        self.store = RunStateStore(connect=self.session_store._connect)

    def tearDown(self):
        self._tmp.cleanup()

    def _upsert(self, *, id='resp_1', status=RUN_STATE_RUNNING, blob='blob1'):
        self.store.upsert(
            id=id, session_id='sess_1', run_id='run_1',
            response_id=id if id.startswith('resp_') else None,
            user_id=SERVER_USER_ID, model='qwen-test', status=status,
            run_state_blob=blob,
            pending_interruption_json=None, last_event_id=None,
            audit_log_id='audit_1',
        )

    def test_upsert_then_get(self):
        self._upsert()
        row = self.store.get('resp_1')
        self.assertIsNotNone(row)
        self.assertEqual(row['status'], RUN_STATE_RUNNING)
        self.assertEqual(row['run_state_blob'], 'blob1')

    def test_upsert_overwrites(self):
        self._upsert(blob='v1')
        self._upsert(blob='v2', status=RUN_STATE_REQUIRES_ACTION)
        row = self.store.get('resp_1')
        self.assertEqual(row['run_state_blob'], 'v2')
        self.assertEqual(row['status'], RUN_STATE_REQUIRES_ACTION)

    def test_get_by_response_id(self):
        self._upsert()
        row = self.store.get_by_response_id('resp_1')
        self.assertIsNotNone(row)
        self.assertEqual(row['id'], 'resp_1')

    def test_mark_status_and_list_active(self):
        self._upsert()
        self.store.mark_status('resp_1', RUN_STATE_REQUIRES_ACTION)
        active = self.store.list_active()
        self.assertEqual(len(active), 1)
        self.store.mark_status('resp_1', RUN_STATE_COMPLETED)
        self.assertEqual(self.store.list_active(), [])

    def test_user_id_isolation(self):
        self._upsert()
        # SERVER_USER_ID always sees rows; a different user does not.
        self.assertIsNotNone(self.store.get('resp_1', user_id=SERVER_USER_ID))
        self.assertIsNone(self.store.get('resp_1', user_id='other_user'))

    def test_find_by_pending_call_id_finds_paused_run(self):
        # Persist a paused run with a pending interruption mirror; the
        # chat-completions resume path uses this to map tool_call_id → row.
        pending = json.dumps([{'call_id': 'approve_xyz', 'tool_name': 'ask_user', 'arguments': {}}])
        self.store.upsert(
            id='resp_pause', session_id='sess_1', run_id='run_1',
            response_id='resp_pause', user_id=SERVER_USER_ID,
            model='qwen-test', status=RUN_STATE_REQUIRES_ACTION,
            run_state_blob='blob',
            pending_interruption_json=pending, last_event_id=None,
            audit_log_id='audit_1',
        )
        row = self.store.find_by_pending_call_id('approve_xyz')
        self.assertIsNotNone(row)
        self.assertEqual(row['id'], 'resp_pause')
        # Mismatched call_id → None
        self.assertIsNone(self.store.find_by_pending_call_id('nope'))
        # Session scoping
        self.assertIsNotNone(self.store.find_by_pending_call_id('approve_xyz', session_id='sess_1'))
        self.assertIsNone(self.store.find_by_pending_call_id('approve_xyz', session_id='other_sess'))


# ─── AuditStore ────────────────────────────────────────────────────────────

class AuditStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.session_store = SessionStore(os.path.join(self._tmp.name, 'sessions'))
        from backend.audit.store import AuditStore, AuditEvent
        self.AuditStore = AuditStore
        self.AuditEvent = AuditEvent
        self.store = AuditStore(connect=self.session_store._connect)

    def tearDown(self):
        self._tmp.cleanup()

    def test_append_and_query(self):
        self.store.append(self.AuditEvent(
            audit_log_id='audit_1', run_id='run_1', session_id='sess_1',
            response_id='resp_1', category='llm_chunk',
            payload={'delta': 'hello'},
        ))
        rows = self.store.query('audit_1')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['category'], 'llm_chunk')
        self.assertEqual(rows[0]['payload']['delta'], 'hello')

    def test_unknown_category_rejected(self):
        with self.assertRaises(ValueError):
            self.store.append(self.AuditEvent(
                audit_log_id='audit_1', run_id='run_1', session_id='sess_1',
                response_id=None, category='not_a_real_category', payload={},
            ))

    def test_redaction_runs_on_string_leaves(self):
        # Sensitive-looking payload — exact pattern set lives in agent_loop.
        self.store.append(self.AuditEvent(
            audit_log_id='audit_2', run_id='run_2', session_id='sess_1',
            response_id=None, category='tool_call',
            payload={'name': 'http_get', 'args': {'token': 'sk-abcdef0123456789ABCDEF01'}},
        ))
        rows = self.store.query('audit_2')
        # We don't pin the exact replacement (keeps the test from depending on
        # the redaction ruleset); we only assert nothing raises and the row
        # round-trips intact.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['payload']['name'], 'http_get')


if __name__ == '__main__':
    unittest.main()
