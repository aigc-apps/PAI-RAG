"""FastAPI-side coordinator for Celery-backed agent runs."""
import os
import uuid
from dataclasses import dataclass

from backend.agent_service import NoRegeneratableAnswerError, ServiceCapacityError, SessionBusyError
from backend.redis_bus import RedisBus
from backend.workspace import WorkspaceViolation
from session_store import SERVER_USER_ID
import settings as config


@dataclass
class RunStream:
    session_id: str
    run_id: str
    stream_from: str = '0-0'
    regenerated_from_run_id: str = ''


class CeleryRunService:
    def __init__(self, store, workspace_manager):
        self.store = store
        self.workspace_manager = workspace_manager
        self.bus = RedisBus()

    def _workspace_for(self, user_id, session_id, cwd=None):
        user_id = user_id or SERVER_USER_ID
        if user_id == SERVER_USER_ID and not getattr(config, 'ENFORCE_WORKSPACE_FOR_SERVER', False):
            return os.path.abspath(cwd or os.getcwd())
        return self.workspace_manager.prepare_session(user_id, session_id, cwd=cwd)

    def start_or_answer(self, session_id, user_id, text, mode='events', cwd=None):
        loaded = self.store.load(session_id, user_id=user_id)
        if loaded is None:
            return None

        if loaded.get('status') == 'waiting_user' and loaded.get('active_run_id'):
            run_id = loaded['active_run_id']
            stream_from = self.bus.last_event_id(run_id)
            answered = self.store.answer_waiting_run(session_id, user_id, run_id, text)
            if not answered:
                latest = self.store.load(session_id, user_id=user_id) or {}
                raise SessionBusyError(session_id, latest.get('status', 'unknown'))
            self.bus.push_answer(run_id, text)
            return RunStream(session_id=session_id, run_id=run_id, stream_from=stream_from)

        run_id = f'run_{uuid.uuid4().hex}'
        workspace_path = loaded.get('workspace_path') or self._workspace_for(user_id, session_id, cwd=cwd)
        result = self.store.try_start_run(
            session_id=session_id,
            user_id=user_id,
            run_id=run_id,
            mode=mode,
            user_text=text,
            workspace_path=workspace_path,
            max_global_runs=int(getattr(config, 'MAX_GLOBAL_RUNS', 0) or 0),
            max_user_runs=int(getattr(config, 'MAX_USER_RUNS', 0) or 0),
        )
        if result['status'] == 'not_found':
            return None
        if result['status'] == 'busy':
            raise SessionBusyError(session_id, result.get('session_status', 'running'))
        if result['status'] == 'capacity':
            raise ServiceCapacityError(result.get('scope', 'global'), result.get('limit', 0))
        if result['status'] == 'workspace_violation':
            raise WorkspaceViolation(result.get('message') or 'workspace_violation')

        try:
            self._enqueue_run(run_id, session_id, user_id, text, mode, cwd)
        except Exception:
            self.store.finish_run(session_id, user_id, run_id, 'failed', error='failed to enqueue celery task')
            raise
        return RunStream(session_id=session_id, run_id=run_id, stream_from='0-0')

    def regenerate_last_answer(self, session_id, user_id, mode='events', cwd=None):
        loaded = self.store.load(session_id, user_id=user_id)
        if loaded is None:
            return None

        run_id = f'run_{uuid.uuid4().hex}'
        workspace_path = loaded.get('workspace_path') or self._workspace_for(user_id, session_id, cwd=cwd)
        result = self.store.try_start_regenerate_run(
            session_id=session_id,
            user_id=user_id,
            run_id=run_id,
            mode=mode,
            workspace_path=workspace_path,
            max_global_runs=int(getattr(config, 'MAX_GLOBAL_RUNS', 0) or 0),
            max_user_runs=int(getattr(config, 'MAX_USER_RUNS', 0) or 0),
        )
        if result['status'] == 'not_found':
            return None
        if result['status'] == 'busy':
            raise SessionBusyError(session_id, result.get('session_status', 'running'))
        if result['status'] == 'capacity':
            raise ServiceCapacityError(result.get('scope', 'global'), result.get('limit', 0))
        if result['status'] == 'workspace_violation':
            raise WorkspaceViolation(result.get('message') or 'workspace_violation')
        if result['status'] == 'no_regeneratable_answer':
            raise NoRegeneratableAnswerError(session_id)

        try:
            self._enqueue_run(run_id, session_id, user_id, result['input_text'], mode, cwd)
        except Exception:
            self.store.finish_run(session_id, user_id, run_id, 'failed', error='failed to enqueue celery task')
            raise
        return RunStream(
            session_id=session_id,
            run_id=run_id,
            stream_from='0-0',
            regenerated_from_run_id=result.get('regenerated_from_run_id') or '',
        )

    def _enqueue_run(self, run_id, session_id, user_id, text, mode, cwd):
        from backend.worker import run_agent_task

        run_agent_task.delay(run_id, session_id, user_id, text, mode, cwd)

    def iter_events(self, run_id, stream_from='0-0'):
        return self.bus.iter_events(run_id, last_id=stream_from)

    def load_run(self, run_id, user_id):
        return self.store.load_run(run_id, user_id=user_id)

    def read_events(self, run_id, user_id, last_id='0-0', block_ms=1000):
        run = self.load_run(run_id, user_id)
        if run is None:
            return None
        events = self.bus.read_events(run_id, last_id=last_id, block_ms=block_ms)
        if events:
            self.store.set_run_last_event(run_id, user_id, events[-1][0])
        return events

    def cancel_run(self, run_id, user_id):
        run = self.load_run(run_id, user_id)
        if run is None:
            return False
        self.bus.cancel(run_id)
        if run.get('status') == 'waiting_user':
            self.bus.push_answer(run_id, '[Cancelled]')
        self.store.request_cancel(run['session_id'], user_id, run_id)
        return True

    def cancel_session(self, session_id, user_id):
        loaded = self.store.load(session_id, user_id=user_id)
        if loaded is None:
            return False
        run_id = loaded.get('active_run_id')
        if not run_id:
            return True
        self.bus.cancel(run_id)
        if loaded.get('status') == 'waiting_user':
            self.bus.push_answer(run_id, '[Cancelled]')
        self.store.request_cancel(session_id, user_id, run_id)
        return True
