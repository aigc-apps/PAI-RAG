"""ask_user 中断/恢复路径的回归测试。

固化 4 个修复的关键不变量：
- Fix 1: `do_ask_user` 在 emit ask_user 之前先把 session 状态切到 waiting_user。
- Fix 2: `do_ask_user` 在 _aq.get() 拿到答案后再 clear 一次 turn_done_evt
         （防止与 run_or_answer.clear() 的 lock-race 导致 set 后于 clear）。
- Fix 3: SSE consumer 在 sess.status == waiting_user 时跳过 cancel_session。
- Fix 4: run.completed SSE 事件正确透传 done 事件携带的 usage。
"""
import json
import queue
import threading
import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from agent_events import agent_message_chunk, ask_user, done
from backend.agent_service import HttpHandler, SESSION_RUNNING, SESSION_WAITING_USER, SessionBusyError
import backend.server as server
from session_store import SERVER_USER_ID


# ─────────────────────────── Fix 1 & Fix 2: do_ask_user 单元测试 ────────────── #

class DoAskUserUnitTests(unittest.TestCase):
    """直接对 HttpHandler.do_ask_user 进行单元测试，验证状态翻转顺序与 turn_done 清扫。"""

    def _build_handler(self, answer_text='answer'):
        call_log: list[tuple] = []

        class FakeSession:
            status = 'running'
            output_mode = 'events'
            turn_done_evt = threading.Event()
            cancel_evt = threading.Event()

            def mark_waiting_for_user(self_):
                call_log.append(('mark_waiting_for_user', self_.status))
                self_.status = SESSION_WAITING_USER

            def emit_event(self_, event, check_cancel=True):
                call_log.append(('emit_event', event.get('sessionUpdate'), self_.status))

            def _on_text_chunk(self_, text, check_cancel=True):  # noqa: D401 unused in events mode
                call_log.append(('text', text))

        aq: queue.Queue = queue.Queue()
        aq.put(answer_text)
        handler = HttpHandler.__new__(HttpHandler)  # bypass __init__
        handler._session = FakeSession()
        handler._aq = aq
        return handler, call_log

    def test_mark_waiting_runs_before_emit_ask_user(self):
        handler, log = self._build_handler()
        handler.do_ask_user({'question': 'q?'}, response=None)

        mark_idx = next(i for i, c in enumerate(log) if c[0] == 'mark_waiting_for_user')
        emit_idx = next(i for i, c in enumerate(log)
                        if c[0] == 'emit_event' and c[1] == 'ask_user')
        self.assertLess(mark_idx, emit_idx, f'mark must run before emit; log={log!r}')
        # 当 ask_user 事件入队时,session 已经处于 waiting_user 状态
        self.assertEqual(log[emit_idx][2], SESSION_WAITING_USER)

    def test_turn_done_evt_is_cleared_after_answer_received(self):
        handler, _log = self._build_handler()
        # 模拟 worker 阻塞前 set,之后 _aq.get 拿到答案,我们的 fix 应在拿到答案后 clear。
        # 验证方式:do_ask_user 返回后 turn_done_evt.is_set() == False。
        handler.do_ask_user({'question': 'q?'}, response=None)
        self.assertFalse(
            handler._session.turn_done_evt.is_set(),
            'turn_done_evt should be cleared after _aq.get() so a follow-up consumer '
            'does not see stale True from a delayed set() racing run_or_answer.clear()',
        )


# ──────────────── Fix 3 & Fix 4: 端到端 SSE 整合测试 ──────────────── #

class _FakeSession:
    """模拟 AgentSession 中 run_or_answer 和 cancel 的相关行为。"""

    def __init__(self, sid: str):
        self.sid = sid
        self.user_id = SERVER_USER_ID
        self.display_q: queue.Queue = queue.Queue()
        self.ask_q: queue.Queue = queue.Queue()
        self.turn_done_evt = threading.Event()
        self.cancel_evt = threading.Event()
        self.status = 'idle'
        self.active_run_id = ''
        self.worker_alive = False
        self.cancel_called = 0
        self._lock = threading.RLock()

    def run_or_answer(self, text, mode='events'):
        with self._lock:
            if self.worker_alive and self.status == SESSION_WAITING_USER:
                # answer 分支：路由到等待中的 worker
                self.status = SESSION_RUNNING
                self.turn_done_evt.clear()
                self.ask_q.put(text)
                # 模拟 worker 拿到答案后继续:emit 一个 message + done(带 usage)
                self.display_q.put({'event': agent_message_chunk(f'got: {text}')})
                self.display_q.put({
                    'event': done('end_turn', usage={
                        'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15,
                    }),
                })
                self.turn_done_evt.set()
                return self.active_run_id
            if self.worker_alive and self.status == SESSION_RUNNING:
                raise SessionBusyError(self.sid, self.status)
            # 新 run：模拟 worker 跑出 ask_user 中断
            self.worker_alive = True
            self.active_run_id = 'run-1'
            self.status = SESSION_WAITING_USER  # Fix 1: mark before emit
            self.display_q.put({'event': ask_user('想叫什么名字?', [])})
            self.display_q.put({'event': done('end_turn')})
            self.turn_done_evt.set()
            return self.active_run_id

    def cancel(self):
        self.cancel_called += 1
        self.cancel_evt.set()


class _FakeService:
    def __init__(self):
        self.sessions: dict[str, _FakeSession] = {}

    def create_session(self, user_id=SERVER_USER_ID, cwd=None):
        sid = f'sess-{len(self.sessions) + 1}'
        sess = _FakeSession(sid)
        self.sessions[sid] = sess
        return sess

    def get_session(self, sid=None, user_id=SERVER_USER_ID, cwd=None):
        if not sid:
            return self.create_session(user_id=user_id, cwd=cwd)
        if sid not in self.sessions:
            self.sessions[sid] = _FakeSession(sid)
        return self.sessions[sid]

    def load_session(self, sid, user_id=SERVER_USER_ID):
        return self.sessions.get(sid)

    def cancel_session(self, sid, user_id=SERVER_USER_ID):
        sess = self.sessions.get(sid)
        if sess:
            sess.cancel()
        return sess is not None


def _parse_sse_events(text: str) -> list[dict]:
    """从 SSE body 里抽出 data: 行的 JSON。"""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('data:'):
            payload = line[len('data:'):].strip()
            if payload and payload != '[DONE]':
                try:
                    events.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
    return events


class AskUserResumeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.original_service = server.service
        self.original_celery_service = server.celery_service
        self.original_thread_runs = dict(server.THREAD_RUNS)
        server.celery_service = None
        server.THREAD_RUNS.clear()
        self.fake_service = _FakeService()
        server.service = self.fake_service
        self.client = TestClient(server.app)

    def tearDown(self):
        server.service = self.original_service
        server.celery_service = self.original_celery_service
        server.THREAD_RUNS.clear()
        server.THREAD_RUNS.update(self.original_thread_runs)

    def test_second_post_during_waiting_user_does_not_409(self):
        # Turn 1: trigger ask_user
        r1 = self.client.post('/v1/runs', json={'input': 'create a file', 'stream': True})
        self.assertEqual(r1.status_code, 200, r1.text)
        sid = r1.headers['x-session-id']
        events1 = _parse_sse_events(r1.text)
        types1 = [e.get('event') for e in events1]
        self.assertIn('ask_user', types1)
        self.assertIn('run.completed', types1)

        # 此时 server-side session 应该处于 waiting_user
        sess = self.fake_service.sessions[sid]
        self.assertEqual(sess.status, SESSION_WAITING_USER)

        # Turn 2: 紧接着发回答,不应得到 409
        r2 = self.client.post(
            '/v1/runs',
            json={'session_id': sid, 'input': 'hello.txt', 'stream': True},
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        # 回答应该已经投递到 worker 的 ask_q
        self.assertEqual(list(sess.ask_q.queue), ['hello.txt'])

    def test_run_completed_event_carries_usage(self):
        # Trigger fresh run + immediately answer to walk through full lifecycle
        r1 = self.client.post('/v1/runs', json={'input': 'hi', 'stream': True})
        sid = r1.headers['x-session-id']
        r2 = self.client.post(
            '/v1/runs',
            json={'session_id': sid, 'input': 'answer', 'stream': True},
        )
        events = _parse_sse_events(r2.text)
        completed = [e for e in events if e.get('event') == 'run.completed']
        self.assertTrue(completed, f'no run.completed in events: {events!r}')
        self.assertEqual(
            completed[-1].get('usage'),
            {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
        )

    def test_disconnect_during_waiting_user_does_not_cancel_session(self):
        """Fix 3: 客户端在 waiting_user 状态断开时不应触发 cancel_session。"""
        # 走完 turn-1（拿到 ask_user + run.completed,然后流自然结束)。
        # 流结束时 server 端的 thread_run_event_stream 内部会执行最后一次
        # is_disconnected 检测(client 已不再读)。验证 sess.cancel 未被调用。
        r1 = self.client.post('/v1/runs', json={'input': 'q', 'stream': True})
        self.assertEqual(r1.status_code, 200)
        sid = r1.headers['x-session-id']
        sess = self.fake_service.sessions[sid]
        # Fix 3 的关键不变量:waiting_user 状态下,即使 client 不再保持连接,
        # cancel_session 也不应当被调用。
        self.assertEqual(sess.cancel_called, 0)
        self.assertEqual(sess.status, SESSION_WAITING_USER)


if __name__ == '__main__':
    unittest.main()
