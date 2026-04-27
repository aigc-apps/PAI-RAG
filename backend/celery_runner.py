"""FastAPI-side coordinator for Celery-backed agent runs."""
import os
import uuid
from dataclasses import dataclass

from backend.agent_service import ServiceCapacityError, SessionBusyError
from backend.redis_bus import RedisBus
from backend.workspace import WorkspaceViolation
from session_store import SERVER_USER_ID

try:
    import config
except ImportError as e:
    raise RuntimeError('config.py not found. Copy config_template.py to config.py first.') from e


@dataclass
class RunStream:
    session_id: str
    run_id: str
    stream_from: str = '0-0'


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

        run_id = uuid.uuid4().hex
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

    def _enqueue_run(self, run_id, session_id, user_id, text, mode, cwd):
        from backend.worker import run_agent_task

        run_agent_task.delay(run_id, session_id, user_id, text, mode, cwd)

    def iter_events(self, run_id, stream_from='0-0'):
        return self.bus.iter_events(run_id, last_id=stream_from)

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
