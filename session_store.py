"""Session persistence backed by memory/sessions_v2.sqlite3."""
import json, os, re, sqlite3, threading
from datetime import datetime


SERVER_USER_ID = '__server__'
ACTIVE_STATUSES = {'running', 'waiting_user'}
DEFAULT_SESSION_TITLE = 'New Task'
SQLITE_JOURNAL_MODES = {'DELETE', 'TRUNCATE', 'PERSIST', 'MEMORY', 'WAL', 'OFF'}


def sqlite_journal_mode():
    try:
        import config
        configured = getattr(config, 'SQLITE_JOURNAL_MODE', '')
    except ImportError:
        configured = ''
    mode = (os.environ.get('SQLITE_JOURNAL_MODE') or configured or 'DELETE').upper()
    return mode if mode in SQLITE_JOURNAL_MODES else 'DELETE'


def session_title_from_messages(ui_messages, current_title=None):
    title = (current_title or '').strip()
    if title and title != DEFAULT_SESSION_TITLE:
        return title
    return next(
        ((m.get('content') or '').strip()[:60] for m in ui_messages if m.get('role') == 'user' and (m.get('content') or '').strip()),
        DEFAULT_SESSION_TITLE,
    )


class SessionStore:
    _SESSION_ID_RE = re.compile(r'^[A-Za-z0-9_-]+$')

    def __init__(self, sessions_dir):
        sessions_dir = os.path.abspath(sessions_dir)
        os.makedirs(sessions_dir, exist_ok=True)
        self._db_path = os.path.join(os.path.dirname(sessions_dir), 'sessions_v2.sqlite3')
        self._locks = {}
        self._locks_guard = threading.Lock()
        self._init_db()

    @property
    def db_path(self):
        return self._db_path

    def save(
        self,
        session_id,
        llm_history,
        ui_messages,
        handler_state=None,
        title=None,
        user_id=SERVER_USER_ID,
        status='idle',
        active_run_id=None,
        workspace_path='',
    ):
        self._validate_session_id(session_id)
        with self._lock_for(session_id):
            old_data = self._load_from_db(session_id, user_id=user_id)
            now = datetime.now().isoformat()
            created_at = old_data.get('created_at', now) if old_data else now
            if ui_messages is None:
                ui_messages = old_data.get('ui_messages', []) if old_data else []
            if title is None:
                title = session_title_from_messages(ui_messages, old_data.get('title') if old_data else None)
            data = {
                'session_id': session_id,
                'user_id': user_id,
                'created_at': created_at,
                'updated_at': now,
                'title': title,
                'llm_history': llm_history,
                'ui_messages': ui_messages,
                'handler_state': handler_state,
                'status': status or 'idle',
                'active_run_id': active_run_id,
                'workspace_path': workspace_path or '',
            }
            self._save_data(data)

    def load(self, session_id, user_id=SERVER_USER_ID):
        self._validate_session_id(session_id)
        with self._lock_for(session_id):
            return self._load_from_db(session_id, user_id=user_id)

    def list_sessions(self, user_id=SERVER_USER_ID):
        with self._connect() as conn:
            rows = conn.execute(
                '''
                SELECT session_id, title, created_at, updated_at, message_count, status, active_run_id
                FROM sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                '''
                ,
                (user_id,),
            ).fetchall()
        return [{
            'session_id': row['session_id'],
            'title': row['title'] or DEFAULT_SESSION_TITLE,
            'created_at': row['created_at'] or '',
            'updated_at': row['updated_at'] or '',
            'message_count': row['message_count'] or 0,
            'status': row['status'] or 'idle',
            'active_run_id': row['active_run_id'] or '',
        } for row in rows]

    def delete(self, session_id, user_id=SERVER_USER_ID):
        self._validate_session_id(session_id)
        with self._lock_for(session_id):
            with self._connect() as conn:
                conn.execute('DELETE FROM runs WHERE session_id = ? AND user_id = ?', (session_id, user_id))
                cur = conn.execute('DELETE FROM sessions WHERE session_id = ? AND user_id = ?', (session_id, user_id))
            return cur.rowcount > 0

    def try_start_run(
        self,
        session_id,
        user_id,
        run_id,
        mode,
        user_text,
        workspace_path='',
        max_global_runs=0,
        max_user_runs=0,
    ):
        self._validate_session_id(session_id)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = self._session_row(conn, session_id, user_id)
            if row is None:
                conn.rollback()
                return {'status': 'not_found'}
            if row['status'] in ACTIVE_STATUSES:
                conn.rollback()
                return {'status': 'busy', 'session_status': row['status'], 'run_id': row['active_run_id'] or ''}

            if max_global_runs:
                active_count = conn.execute(
                    'SELECT COUNT(*) AS n FROM sessions WHERE status IN (?, ?)',
                    ('running', 'waiting_user'),
                ).fetchone()['n']
                if active_count >= max_global_runs:
                    conn.rollback()
                    return {'status': 'capacity', 'scope': 'global', 'limit': max_global_runs}
            if max_user_runs:
                user_active_count = conn.execute(
                    'SELECT COUNT(*) AS n FROM sessions WHERE user_id = ? AND status IN (?, ?)',
                    (user_id, 'running', 'waiting_user'),
                ).fetchone()['n']
                if user_active_count >= max_user_runs:
                    conn.rollback()
                    return {'status': 'capacity', 'scope': 'user', 'limit': max_user_runs}

            ui_messages = self._json_list(row['ui_messages_json'])
            ui_messages.append({'role': 'user', 'content': user_text})
            ui_messages.append({'role': 'assistant', 'content': '', 'events': []})
            title = session_title_from_messages(ui_messages, row['title'])
            conn.execute(
                '''
                UPDATE sessions
                SET status = ?, active_run_id = ?, workspace_path = ?, ui_messages_json = ?,
                    message_count = ?, title = ?, updated_at = ?
                WHERE session_id = ? AND user_id = ?
                ''',
                (
                    'running',
                    run_id,
                    workspace_path or row['workspace_path'] or '',
                    json.dumps(ui_messages, ensure_ascii=False, default=str),
                    len(ui_messages),
                    title,
                    now,
                    session_id,
                    user_id,
                ),
            )
            conn.execute(
                '''
                INSERT INTO runs (run_id, session_id, user_id, mode, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (run_id, session_id, user_id, mode, 'queued', now, now),
            )
            conn.commit()
        return {'status': 'started', 'run_id': run_id}

    def answer_waiting_run(self, session_id, user_id, run_id, answer):
        self._validate_session_id(session_id)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = self._session_row(conn, session_id, user_id)
            if row is None or row['status'] != 'waiting_user' or row['active_run_id'] != run_id:
                conn.rollback()
                return False
            ui_messages = self._json_list(row['ui_messages_json'])
            ui_messages.append({'role': 'user', 'content': answer})
            conn.execute(
                '''
                UPDATE sessions
                SET status = ?, ui_messages_json = ?, message_count = ?, updated_at = ?
                WHERE session_id = ? AND user_id = ?
                ''',
                (
                    'running',
                    json.dumps(ui_messages, ensure_ascii=False, default=str),
                    len(ui_messages),
                    now,
                    session_id,
                    user_id,
                ),
            )
            conn.execute(
                'UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ? AND user_id = ?',
                ('running', now, run_id, user_id),
            )
            conn.commit()
        return True

    def mark_waiting_user(self, session_id, user_id, run_id):
        self._validate_session_id(session_id)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                '''
                UPDATE sessions
                SET status = ?, active_run_id = ?, updated_at = ?
                WHERE session_id = ? AND user_id = ? AND active_run_id = ?
                ''',
                ('waiting_user', run_id, now, session_id, user_id, run_id),
            )
            conn.execute(
                'UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ? AND user_id = ?',
                ('waiting_user', now, run_id, user_id),
            )

    def set_run_status(self, run_id, user_id, status):
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                'UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ? AND user_id = ?',
                (status, now, run_id, user_id),
            )

    def finish_run(self, session_id, user_id, run_id, status, error=''):
        self._validate_session_id(session_id)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                '''
                UPDATE sessions
                SET status = ?, active_run_id = NULL, updated_at = ?
                WHERE session_id = ? AND user_id = ? AND active_run_id = ?
                ''',
                (status, now, session_id, user_id, run_id),
            )
            conn.execute(
                'UPDATE runs SET status = ?, error = ?, updated_at = ?, finished_at = ? WHERE run_id = ? AND user_id = ?',
                (status, error or '', now, now, run_id, user_id),
            )

    def save_run_snapshot(
        self,
        session_id,
        user_id,
        run_id,
        llm_history,
        ui_messages,
        handler_state=None,
        status='running',
        active_run_id=None,
        workspace_path='',
    ):
        self._validate_session_id(session_id)
        now = datetime.now().isoformat()
        ui_messages = ui_messages or []
        active_run_id = active_run_id if active_run_id is not None else run_id
        title = session_title_from_messages(ui_messages)
        with self._connect() as conn:
            cur = conn.execute(
                '''
                UPDATE sessions
                SET updated_at = ?,
                    title = CASE
                        WHEN title IS NULL OR trim(title) = '' OR title = ? THEN ?
                        ELSE title
                    END,
                    llm_history_json = ?,
                    ui_messages_json = ?,
                    handler_state_json = ?,
                    message_count = ?,
                    status = ?,
                    active_run_id = ?,
                    workspace_path = ?
                WHERE session_id = ? AND user_id = ? AND active_run_id = ?
                ''',
                (
                    now,
                    DEFAULT_SESSION_TITLE,
                    title,
                    json.dumps(llm_history or [], ensure_ascii=False, default=str),
                    json.dumps(ui_messages, ensure_ascii=False, default=str),
                    json.dumps(handler_state, ensure_ascii=False, default=str),
                    len(ui_messages),
                    status or 'running',
                    active_run_id,
                    workspace_path or '',
                    session_id,
                    user_id,
                    run_id,
                ),
            )
        return cur.rowcount > 0

    def request_cancel(self, session_id, user_id, run_id):
        self._validate_session_id(session_id)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                '''
                UPDATE sessions
                SET status = ?, updated_at = ?
                WHERE session_id = ? AND user_id = ? AND active_run_id = ?
                ''',
                ('cancelled', now, session_id, user_id, run_id),
            )
            conn.execute(
                'UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ? AND user_id = ?',
                ('cancelled', now, run_id, user_id),
            )

    def owner_for(self, session_id):
        self._validate_session_id(session_id)
        with self._connect() as conn:
            row = conn.execute(
                'SELECT user_id FROM sessions WHERE session_id = ?',
                (session_id,),
            ).fetchone()
        return row['user_id'] if row else None

    def session_exists(self, session_id):
        self._validate_session_id(session_id)
        with self._connect() as conn:
            row = conn.execute(
                'SELECT 1 FROM sessions WHERE session_id = ?',
                (session_id,),
            ).fetchone()
        return row is not None

    def _validate_session_id(self, session_id):
        if not session_id or not self._SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(f'Invalid session_id: {session_id!r}')

    def _lock_for(self, session_id):
        with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.RLock()
                self._locks[session_id] = lock
            return lock

    def _connect(self):
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(f'PRAGMA journal_mode={sqlite_journal_mode()}')
            conn.execute('PRAGMA busy_timeout=30000')
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    llm_history_json TEXT NOT NULL,
                    ui_messages_json TEXT NOT NULL,
                    handler_state_json TEXT,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'idle',
                    active_run_id TEXT,
                    workspace_path TEXT
                )
                '''
            )
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                )
                '''
            )
            columns = {
                row['name']
                for row in conn.execute('PRAGMA table_info(sessions)').fetchall()
            }
            if 'user_id' not in columns:
                conn.execute('ALTER TABLE sessions ADD COLUMN user_id TEXT')
            if 'status' not in columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN status TEXT NOT NULL DEFAULT 'idle'")
            if 'active_run_id' not in columns:
                conn.execute('ALTER TABLE sessions ADD COLUMN active_run_id TEXT')
            if 'workspace_path' not in columns:
                conn.execute('ALTER TABLE sessions ADD COLUMN workspace_path TEXT')
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC)'
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC)'
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_runs_session_updated ON runs(session_id, updated_at DESC)'
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_runs_user_status ON runs(user_id, status)'
            )

    def _session_row(self, conn, session_id, user_id):
        return conn.execute(
            '''
            SELECT session_id, user_id, created_at, updated_at, title,
                   llm_history_json, ui_messages_json, handler_state_json,
                   status, active_run_id, workspace_path
            FROM sessions
            WHERE session_id = ? AND user_id = ?
            ''',
            (session_id, user_id),
        ).fetchone()

    @staticmethod
    def _json_list(raw):
        try:
            value = json.loads(raw or '[]')
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []

    def _save_data(self, data):
        ui_messages = data.get('ui_messages') or []
        user_id = data.get('user_id') or SERVER_USER_ID
        with self._connect() as conn:
            owner = conn.execute(
                'SELECT user_id FROM sessions WHERE session_id = ?',
                (data['session_id'],),
            ).fetchone()
            if owner and owner['user_id'] != user_id:
                raise PermissionError('Session belongs to another user')
            conn.execute(
                '''
                INSERT INTO sessions (
                    session_id, user_id, created_at, updated_at, title,
                    llm_history_json, ui_messages_json, handler_state_json,
                    message_count, status, active_run_id, workspace_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    title = excluded.title,
                    llm_history_json = excluded.llm_history_json,
                    ui_messages_json = excluded.ui_messages_json,
                    handler_state_json = excluded.handler_state_json,
                    message_count = excluded.message_count,
                    status = excluded.status,
                    active_run_id = excluded.active_run_id,
                    workspace_path = excluded.workspace_path
                ''',
                (
                    data['session_id'],
                    user_id,
                    data.get('created_at') or datetime.now().isoformat(),
                    data.get('updated_at') or datetime.now().isoformat(),
                    data.get('title') or DEFAULT_SESSION_TITLE,
                    json.dumps(data.get('llm_history') or [], ensure_ascii=False, default=str),
                    json.dumps(ui_messages, ensure_ascii=False, default=str),
                    json.dumps(data.get('handler_state'), ensure_ascii=False, default=str),
                    len(ui_messages),
                    data.get('status') or 'idle',
                    data.get('active_run_id'),
                    data.get('workspace_path') or '',
                ),
            )

    def _load_from_db(self, session_id, user_id=SERVER_USER_ID):
        with self._connect() as conn:
            row = conn.execute(
                '''
                SELECT session_id, user_id, created_at, updated_at, title,
                       llm_history_json, ui_messages_json, handler_state_json,
                       status, active_run_id, workspace_path
                FROM sessions
                WHERE session_id = ? AND user_id = ?
                ''',
                (session_id, user_id),
            ).fetchone()
        if row is None:
            return None
        try:
            handler_state = json.loads(row['handler_state_json'])
            return {
                'session_id': row['session_id'],
                'user_id': row['user_id'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'title': row['title'],
                'llm_history': json.loads(row['llm_history_json']),
                'ui_messages': json.loads(row['ui_messages_json']),
                'handler_state': handler_state,
                'status': row['status'] or 'idle',
                'active_run_id': row['active_run_id'] or '',
                'workspace_path': row['workspace_path'] or '',
            }
        except (TypeError, json.JSONDecodeError) as e:
            print(f'[Warn] session {session_id} corrupt in sqlite: {e}')
            return None
