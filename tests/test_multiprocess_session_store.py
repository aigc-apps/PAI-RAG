"""Stress test that two processes (FastAPI + Celery worker shape) can share a
single SQLite-backed `SessionStore` without deadlocking or surfacing
'database is locked' errors.

The historical EAS failure was: server process polling Redis stream and
worker process streaming token snapshots both wrote to `runs`/`sessions`
through one OSS-FUSE-backed SQLite file. Layer 0 (per-conn PRAGMAs +
in-process serializer + retry) and Layers 1/3/4 (move hottest writes off
SQLite) together should make this test pass on any POSIX file system.

We can't realistically simulate OSS-FUSE here, but we can pound the local
file with concurrent processes and assert that no `OperationalError`
escapes the store layer, all snapshots land, and run state is consistent.
"""
import multiprocessing as mp
import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _worker_proc(db_dir, session_id, run_id, errors_q):
    from session_store import SERVER_USER_ID, SessionStore

    store = SessionStore(os.path.join(db_dir, 'sessions'))
    try:
        for i in range(200):
            ok = store.save_run_snapshot(
                session_id=session_id,
                user_id=SERVER_USER_ID,
                run_id=run_id,
                llm_history=[
                    {'role': 'user', 'content': 'q'},
                    {'role': 'assistant', 'content': f'tok-{i}'},
                ],
                ui_messages=[
                    {'role': 'user', 'content': 'q'},
                    {'role': 'assistant', 'content': f'tok-{i}', 'events': []},
                ],
                handler_state={'tick': i},
                status='running',
                active_run_id=run_id,
                workspace_path='/tmp/ws',
            )
            if not ok:
                errors_q.put(f'snapshot {i} did not match active run')
                return
    except Exception as e:
        errors_q.put(f'worker exception: {type(e).__name__}: {e}')
        return
    errors_q.put('OK')


def _server_proc(db_dir, session_id, run_id, errors_q):
    from session_store import SERVER_USER_ID, SessionStore

    store = SessionStore(os.path.join(db_dir, 'sessions'))
    try:
        for _ in range(60):
            store.load(session_id, user_id=SERVER_USER_ID)
            store.load_run(run_id, user_id=SERVER_USER_ID)
            store.list_sessions(user_id=SERVER_USER_ID)
            time.sleep(0.005)
    except Exception as e:
        errors_q.put(f'server exception: {type(e).__name__}: {e}')
        return
    errors_q.put('OK')


class MultiprocessSessionStoreTests(unittest.TestCase):
    def test_concurrent_worker_and_server_no_lock_errors(self):
        ctx = mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as root:
            from session_store import SERVER_USER_ID, SessionStore

            store = SessionStore(os.path.join(root, 'sessions'))
            store.save(
                'sX',
                llm_history=[],
                ui_messages=[{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': '', 'events': []}],
                status='running',
                active_run_id='rX',
            )
            store.create_run_record('sX', SERVER_USER_ID, 'rX', mode='events', status='running')

            errors = ctx.Queue()
            workers = [
                ctx.Process(target=_worker_proc, args=(root, 'sX', 'rX', errors)),
                ctx.Process(target=_server_proc, args=(root, 'sX', 'rX', errors)),
            ]
            t0 = time.time()
            for p in workers:
                p.start()
            for p in workers:
                p.join(timeout=60)
                self.assertFalse(p.is_alive(), 'process hung')
            elapsed = time.time() - t0
            self.assertLess(elapsed, 30, f'multiprocess store run took too long: {elapsed:.1f}s')

            results = []
            while not errors.empty():
                results.append(errors.get_nowait())
            for msg in results:
                self.assertEqual(msg, 'OK', f'process reported: {msg}')

            run = store.load_run('rX', user_id=SERVER_USER_ID)
            self.assertIsNotNone(run)
            sess = store.load('sX', user_id=SERVER_USER_ID)
            self.assertEqual(sess['active_run_id'], 'rX')
            self.assertEqual(sess['handler_state']['tick'], 199)


if __name__ == '__main__':
    unittest.main()
