import os
import sys
import threading
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.memory_scope import memory_scope_for, read_index  # noqa: E402
from backend.tool_schemas import main_tools_schema  # noqa: E402
from backend.workspace import WorkspaceManager  # noqa: E402
from session_store import SERVER_USER_ID, SessionStore  # noqa: E402
from skill_manager import (  # noqa: E402
    get_skills_prompt,
    get_use_skill_schema,
    scan_skills,
)
import settings as config  # noqa: E402


TOOLS_SCHEMA = main_tools_schema()
SYS_PROMPT_BASE = open(os.path.join(ROOT, 'prompts', 'sys_prompt.txt'), encoding='utf-8').read()
SKILLS = scan_skills(os.path.join(ROOT, 'skills'))
if SKILLS:
    TOOLS_SCHEMA.append(get_use_skill_schema())


from backend.agents_sdk.lifecycle import (  # noqa: E402
    ACTIVE_SESSION_STATUSES,
    SESSION_CANCELLED,
    SESSION_IDLE,
)


class SessionBusyError(RuntimeError):
    def __init__(self, session_id, status):
        self.session_id = session_id
        self.status = status
        super().__init__(f'session_busy: session {session_id} is {status}')


class NoRegeneratableAnswerError(RuntimeError):
    def __init__(self, session_id):
        self.session_id = session_id
        super().__init__(f'no_regeneratable_answer: session {session_id} has no completed answer to regenerate')


class ServiceCapacityError(RuntimeError):
    def __init__(self, scope, limit):
        self.scope = scope
        self.limit = limit
        super().__init__(f'capacity_exceeded: {scope} active run limit {limit} reached')


def long_term_memory_enabled(user_id=SERVER_USER_ID):
    return (user_id or SERVER_USER_ID) == SERVER_USER_ID or getattr(config, 'ENABLE_LONG_TERM_MEMORY_FOR_USERS', False)


def handler_memory_scope(user_id=SERVER_USER_ID):
    user_id = user_id or SERVER_USER_ID
    if user_id == SERVER_USER_ID or long_term_memory_enabled(user_id):
        return memory_scope_for(ROOT, user_id)
    if getattr(config, 'ENABLE_SHARED_MEMORY_FOR_USERS', False):
        return memory_scope_for(ROOT, SERVER_USER_ID)
    return memory_scope_for(ROOT, user_id)


def build_system_prompt(user_id=SERVER_USER_ID):
    user_id = user_id or SERVER_USER_ID
    notice = ''
    if user_id == SERVER_USER_ID:
        scope = memory_scope_for(ROOT, user_id)
        idx = read_index(scope)
        notice = f'\n[MEMORY SCOPE] Service memory root: {scope.root}\n'
    elif long_term_memory_enabled(user_id):
        scope = memory_scope_for(ROOT, user_id)
        idx = read_index(scope)
        notice = f'\n[MEMORY SCOPE] User-private memory root: {scope.root}\n'
    elif getattr(config, 'ENABLE_SHARED_MEMORY_FOR_USERS', False):
        scope = memory_scope_for(ROOT, SERVER_USER_ID)
        idx = read_index(scope)
        notice = '\n[MEMORY SCOPE] Shared service memory is readable for this user; user memory updates are disabled.\n'
    else:
        idx = '(empty)'
        notice = '\n[MEMORY SCOPE] Long-term memory is disabled for this user.\n'
    return SYS_PROMPT_BASE + notice + '\n' + idx + get_skills_prompt(SKILLS)


def flatten_message_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get('type') == 'text':
                parts.append(block.get('text', ''))
        return '\n'.join(p for p in parts if p)
    return '' if content is None else str(content)


def last_user_text(messages):
    for msg in reversed(messages or []):
        if msg.get('role') == 'user':
            return flatten_message_content(msg.get('content'))
    return ''


class AgentSession:
    """Slim data record around a session_id. The agent loop itself runs
    inside the SDK runner (``backend.agents_sdk``); this class only
    persists workspace + status metadata so HTTP handlers can look up the
    session by id and the SQLite session row stays in sync.
    """

    def __init__(self, service, sid, user_id=SERVER_USER_ID, cwd=None):
        self.service = service
        self.sid = sid
        self.user_id = user_id
        self.cwd, self.workspace_root, self.readonly_roots = self.service.prepare_workspace(user_id, sid, cwd)
        self.workspace_path = self.workspace_root or self.cwd
        self.memory_scope = handler_memory_scope(user_id)
        self.ui_msgs = []
        self.status = SESSION_IDLE
        self.active_run_id = ''
        self.exit_reason = None
        self._lock = threading.RLock()

    def restore_from(self, loaded):
        if not loaded:
            return
        self.ui_msgs = loaded.get('ui_messages', []) or []
        loaded_status = loaded.get('status') or SESSION_IDLE
        self.status = SESSION_IDLE if loaded_status in ACTIVE_SESSION_STATUSES else loaded_status
        self.active_run_id = ''
        if loaded.get('workspace_path') and self.workspace_root:
            self.workspace_path = loaded.get('workspace_path')
            self.workspace_root = self.workspace_path
            self.cwd = self.workspace_path
            os.makedirs(self.workspace_path, exist_ok=True)

    def save(self):
        with self._lock:
            ui_msgs = list(self.ui_msgs)
        self.service.store.save(
            session_id=self.sid,
            user_id=self.user_id,
            llm_history=[],
            ui_messages=ui_msgs,
            handler_state=None,
            status=self.status,
            active_run_id=self.active_run_id,
            workspace_path=self.workspace_path,
        )

    def cancel(self):
        # Best-effort: flip the session row to cancelled. The actual run is
        # owned by the SDK runner, which has its own cancellation path via
        # ``RunState`` cleanup; there's nothing in-process here to interrupt.
        with self._lock:
            if self.status in ACTIVE_SESSION_STATUSES:
                self.status = SESSION_CANCELLED
        self.save()

    def is_running(self):
        # The SDK runner owns run lifetime now; the session row only reports
        # the last persisted status, so an in-flight SDK run will *not*
        # appear as running here. Callers that need live run state should
        # query ``backend.agents_sdk.run_state_store``.
        return False


class AgentService:
    def __init__(self):
        self.store = SessionStore(os.path.join(ROOT, 'memory', 'sessions'))
        workspace_root = getattr(config, 'WORKSPACE_ROOT', os.path.join(ROOT, 'workspaces'))
        self.workspace_manager = WorkspaceManager(workspace_root)
        self._sessions = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(user_id, sid):
        return (user_id or SERVER_USER_ID, sid)

    def prepare_workspace(self, user_id, sid, cwd=None):
        user_id = user_id or SERVER_USER_ID
        enforce_server_workspace = getattr(config, 'ENFORCE_WORKSPACE_FOR_SERVER', False)
        if user_id == SERVER_USER_ID and not enforce_server_workspace:
            return os.path.abspath(cwd or ROOT), None, []
        workspace_cwd = self.workspace_manager.prepare_session(user_id, sid, cwd=cwd)
        return workspace_cwd, self.workspace_manager.session_root(user_id, sid), []

    def create_session(self, user_id=SERVER_USER_ID, cwd=None):
        sid = str(uuid.uuid4())
        user_id = user_id or SERVER_USER_ID
        sess = AgentSession(self, sid, user_id=user_id, cwd=cwd)
        with self._lock:
            self._sessions[self._key(user_id, sid)] = sess
        sess.save()
        return sess

    def get_session(self, sid=None, user_id=SERVER_USER_ID, cwd=None):
        user_id = user_id or SERVER_USER_ID
        if not sid:
            return self.create_session(user_id=user_id, cwd=cwd)
        key = self._key(user_id, sid)
        with self._lock:
            sess = self._sessions.get(key)
            if sess:
                return sess
            sess = AgentSession(self, sid, user_id=user_id, cwd=cwd)
            loaded = self.store.load(sid, user_id=user_id)
            if loaded is not None:
                sess.restore_from(loaded)
            elif self.store.session_exists(sid):
                return None
            self._sessions[key] = sess
            if loaded is None:
                sess.save()
            return sess

    def load_session(self, sid, user_id=SERVER_USER_ID):
        user_id = user_id or SERVER_USER_ID
        key = self._key(user_id, sid)
        with self._lock:
            sess = self._sessions.get(key)
        loaded = self.store.load(sid, user_id=user_id)
        if loaded is None:
            return None
        if sess is None:
            sess = AgentSession(self, sid, user_id=user_id)
            sess.restore_from(loaded)
            with self._lock:
                self._sessions[key] = sess
        return sess

    def list_sessions(self, user_id=SERVER_USER_ID):
        user_id = user_id or SERVER_USER_ID
        rows = self.store.list_sessions(user_id=user_id)
        with self._lock:
            running = {
                sid: sess.is_running()
                for (owner_id, sid), sess in self._sessions.items()
                if owner_id == user_id
            }
        for row in rows:
            row['running'] = running.get(row['session_id'], False) or row.get('status') in ACTIVE_SESSION_STATUSES
        return rows

    def delete_session(self, sid, user_id=SERVER_USER_ID):
        user_id = user_id or SERVER_USER_ID
        key = self._key(user_id, sid)
        with self._lock:
            sess = self._sessions.pop(key, None)
        if sess:
            sess.cancel()
        return self.store.delete(sid, user_id=user_id)

    def cancel_session(self, sid, user_id=SERVER_USER_ID):
        user_id = user_id or SERVER_USER_ID
        with self._lock:
            sess = self._sessions.get(self._key(user_id, sid))
        if not sess:
            return False
        sess.cancel()
        return True

    def regenerate_session(self, sid, user_id=SERVER_USER_ID):
        """Trim the trailing assistant turn off ``ui_messages`` and return the
        last user prompt so callers can re-issue it through the SDK runner.

        Returns ``{'session_id', 'user_text'}`` on success, ``None`` if the
        session doesn't exist, and raises :class:`NoRegeneratableAnswerError`
        when the tail isn't a ``user → assistant`` pair to regenerate.
        """
        user_id = user_id or SERVER_USER_ID
        sess = self.load_session(sid, user_id=user_id)
        if sess is None:
            return None
        with sess._lock:
            ui_msgs = list(sess.ui_msgs)
        if len(ui_msgs) < 2 or ui_msgs[-1].get('role') != 'assistant':
            raise NoRegeneratableAnswerError(sid)
        user_index = len(ui_msgs) - 2
        if ui_msgs[user_index].get('role') != 'user':
            raise NoRegeneratableAnswerError(sid)
        user_text = (ui_msgs[user_index].get('content') or '').strip()
        if not user_text:
            raise NoRegeneratableAnswerError(sid)
        # Drop the trailing assistant turn so the next /v1/responses run
        # rebuilds it. Don't drop the user message — the SDK runner needs
        # the same prompt as input and will re-append the user turn itself
        # via its normal save path.
        with sess._lock:
            sess.ui_msgs = ui_msgs[:user_index]
        sess.save()
        return {'session_id': sid, 'user_text': user_text}

    def pending_hitl_for_session(self, sid, user_id=SERVER_USER_ID):
        """Return ``{response_id, call_id, tool_name, question, ...}`` if the
        session has a paused SDK run awaiting input, else ``None``.

        Lets the frontend recover HITL state across page reloads — without
        this, a refresh during ``ask_user`` orphans the answer flow.
        """
        import json as _json

        user_id = user_id or SERVER_USER_ID
        try:
            with self.store._connect() as conn:
                row = conn.execute(
                    """
                    SELECT response_id, run_id, pending_interruption_json
                    FROM agent_run_states
                    WHERE session_id = ? AND user_id = ? AND status = ?
                    ORDER BY last_active_at DESC
                    LIMIT 1
                    """,
                    (sid, user_id, 'requires_action'),
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        try:
            interruptions = _json.loads(row['pending_interruption_json'] or '[]')
        except (ValueError, TypeError):
            interruptions = []
        if not interruptions:
            return None
        first = interruptions[0]
        args = first.get('arguments') or {}
        if isinstance(args, str):
            try:
                args = _json.loads(args)
            except (ValueError, TypeError):
                args = {}
        return {
            'response_id': row['response_id'],
            'run_id': row['run_id'],
            'call_id': first.get('call_id', ''),
            'tool_name': first.get('tool_name', ''),
            'question': args.get('question'),
            'candidates': args.get('candidates'),
        }
