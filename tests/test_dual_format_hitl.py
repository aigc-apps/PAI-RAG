"""Responses HITL resume and cross-device safety."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from session_store import SERVER_USER_ID, SessionStore
from backend.agents_sdk.hitl import (
    InterruptionEnvelope,
    ResumePayload, interruption_from_sdk_item,
)
from backend.agents_sdk.lifecycle import (
    RUN_STATE_COMPLETED, RUN_STATE_REQUIRES_ACTION,
)
from backend.agents_sdk.run_state_store import RunStateStore


def _approval_item(call_id, tool_name='ask_user', args='{"question":"Pick A or B?"}'):
    """Mirror the public surface of agents.run_items.ToolApprovalItem."""
    cls = type('ToolApprovalItem', (), {})
    inst = cls()
    inst.id = call_id
    inst.tool_name = tool_name
    inst.arguments = args
    inst.raw_item = {'call_id': call_id, 'name': tool_name, 'arguments': args}
    return inst


class _PausedRun:
    """Test scaffolding: stage a ``requires_action`` row in the run-state
    store with a pending interruption mirror, exactly the shape the runner
    persists when SDK emits a ``ToolApprovalItem``.
    """

    def __init__(self, store, *, response_id='resp_pause', session_id='sess_1',
                 run_id='run_1', call_id='approve_xyz', user_id=SERVER_USER_ID,
                 question='Pick A or B?'):
        self.store = store
        self.response_id = response_id
        self.session_id = session_id
        self.run_id = run_id
        self.call_id = call_id
        self.user_id = user_id
        self.envelope = InterruptionEnvelope(
            call_id=call_id, tool_name='ask_user',
            arguments={'question': question}, is_input_request=True,
        )
        pending_mirror = json.dumps([{
            'call_id': call_id, 'tool_name': 'ask_user',
            'arguments': {'question': question},
        }])
        store.upsert(
            id=response_id, session_id=session_id, run_id=run_id,
            response_id=response_id, user_id=user_id, model='qwen-test',
            status=RUN_STATE_REQUIRES_ACTION,
            run_state_blob='{"opaque":"blob"}',
            pending_interruption_json=pending_mirror, last_event_id=None,
            audit_log_id='audit_pause',
        )


class ResponsesHitlTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.session_store = SessionStore(os.path.join(self._tmp.name, 'sessions'))
        self.run_state_store = RunStateStore(connect=self.session_store._connect)

    def tearDown(self):
        self._tmp.cleanup()

    def test_envelope_serializes_to_responses_wire(self):
        env = interruption_from_sdk_item(_approval_item('abc'))
        responses_wire = env.serialize('responses')
        self.assertEqual(responses_wire['call_id'], 'abc')
        self.assertEqual(responses_wire['name'], 'ask_user')
        self.assertEqual(json.loads(responses_wire['arguments']),
                         {'question': env.arguments['question']})

    def test_responses_resume_locates_and_carries_answer(self):
        paused = _PausedRun(self.run_state_store)
        # Mimic ``POST /v1/responses`` body with previous_response_id +
        # function_call_output.
        items = [{
            'type': 'function_call_output',
            'call_id': paused.call_id, 'output': 'A',
        }]
        rp = ResumePayload.from_responses_input(items)
        self.assertEqual(rp.call_id, paused.call_id)
        self.assertEqual(rp.answer, 'A')
        self.assertTrue(rp.approve)
        # Server look-up by previous_response_id (the responses wire path).
        row = self.run_state_store.get(paused.response_id)
        self.assertIsNotNone(row)
        self.assertEqual(row['status'], RUN_STATE_REQUIRES_ACTION)


class CrossDeviceResumeTests(unittest.TestCase):
    """Device A pauses → device B resumes via ``previous_response_id``;
    device A's retry must NOT silently succeed since the run already moved
    on. The HTTP layer maps a non-``requires_action`` row to a 410.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.session_store = SessionStore(os.path.join(self._tmp.name, 'sessions'))
        self.store = RunStateStore(connect=self.session_store._connect)

    def tearDown(self):
        self._tmp.cleanup()

    def _resume_or_raise(self, response_id, *, user_id=SERVER_USER_ID):
        """Mirror the gate ``backend/agents_sdk/runner._resume_run`` enforces:
        only ``requires_action`` rows are resumable.
        """
        row = self.store.get(response_id, user_id=user_id)
        if row is None:
            raise LookupError(f'run_state not found: {response_id}')
        if row['status'] != RUN_STATE_REQUIRES_ACTION:
            raise LookupError(f'run not resumable: status={row["status"]}')
        return row

    def test_device_b_resumes_then_device_a_sees_gone(self):
        paused = _PausedRun(self.store, response_id='resp_xd', call_id='approve_xd')

        # Device B reads the row by previous_response_id and resumes — this
        # is what ``stream_responses_run(previous_response_id=...)`` does.
        row_b = self._resume_or_raise(paused.response_id)
        self.assertEqual(row_b['id'], paused.response_id)

        # Resuming flips the row out of requires_action; the runner writes
        # ``run_state_blob = final_state.to_string()`` and ``status = completed``.
        self.store.upsert(
            id=paused.response_id, session_id=paused.session_id,
            run_id=paused.run_id, response_id=paused.response_id,
            user_id=paused.user_id, model='qwen-test',
            status=RUN_STATE_COMPLETED, run_state_blob='{"opaque":"final"}',
            pending_interruption_json=None, last_event_id=None,
            audit_log_id='audit_pause',
        )

        # Device A's retry must fail — state is no longer requires_action.
        with self.assertRaises(LookupError) as ctx:
            self._resume_or_raise(paused.response_id)
        self.assertIn('not resumable', str(ctx.exception))

    def test_other_user_cannot_resume(self):
        paused = _PausedRun(self.store, response_id='resp_acl', call_id='approve_acl',
                            user_id='alice')
        # Bob shouldn't see Alice's paused run.
        with self.assertRaises(LookupError):
            self._resume_or_raise(paused.response_id, user_id='bob')

    def test_unknown_response_id_raises_lookup_error(self):
        with self.assertRaises(LookupError):
            self._resume_or_raise('resp_does_not_exist')


if __name__ == '__main__':
    unittest.main()
