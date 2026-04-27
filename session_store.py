"""Session persistence backed by memory/sessions.sqlite3."""
import json, os, re, sqlite3, threading
from datetime import datetime


class SessionStore:
    _SESSION_ID_RE = re.compile(r'^[A-Za-z0-9_-]+$')

    def __init__(self, sessions_dir):
        sessions_dir = os.path.abspath(sessions_dir)
        os.makedirs(sessions_dir, exist_ok=True)
        self._db_path = os.path.join(os.path.dirname(sessions_dir), 'sessions.sqlite3')
        self._locks = {}
        self._locks_guard = threading.Lock()
        self._init_db()

    def save(self, session_id, llm_history, ui_messages, handler_state=None, title=None):
        self._validate_session_id(session_id)
        with self._lock_for(session_id):
            old_data = self._load_from_db(session_id)
            now = datetime.now().isoformat()
            created_at = old_data.get('created_at', now) if old_data else now
            if ui_messages is None:
                ui_messages = old_data.get('ui_messages', []) if old_data else []
            if title is None:
                title = next(
                    (m['content'][:60] for m in ui_messages if m.get('role') == 'user'),
                    'New Chat',
                )
            data = {
                'session_id': session_id,
                'created_at': created_at,
                'updated_at': now,
                'title': title,
                'llm_history': llm_history,
                'ui_messages': ui_messages,
                'handler_state': handler_state,
            }
            self._save_data(data)

    def load(self, session_id):
        self._validate_session_id(session_id)
        with self._lock_for(session_id):
            return self._load_from_db(session_id)

    def list_sessions(self):
        with self._connect() as conn:
            rows = conn.execute(
                '''
                SELECT session_id, title, created_at, updated_at, message_count
                FROM sessions
                ORDER BY updated_at DESC
                '''
            ).fetchall()
        return [{
            'session_id': row['session_id'],
            'title': row['title'] or 'New Chat',
            'created_at': row['created_at'] or '',
            'updated_at': row['updated_at'] or '',
            'message_count': row['message_count'] or 0,
        } for row in rows]

    def delete(self, session_id):
        self._validate_session_id(session_id)
        with self._lock_for(session_id):
            with self._connect() as conn:
                cur = conn.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
            return cur.rowcount > 0

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
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=30000')
            conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    llm_history_json TEXT NOT NULL,
                    ui_messages_json TEXT NOT NULL,
                    handler_state_json TEXT,
                    message_count INTEGER NOT NULL DEFAULT 0
                )
                '''
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC)'
            )

    def _save_data(self, data):
        ui_messages = data.get('ui_messages') or []
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO sessions (
                    session_id, created_at, updated_at, title,
                    llm_history_json, ui_messages_json, handler_state_json,
                    message_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    title = excluded.title,
                    llm_history_json = excluded.llm_history_json,
                    ui_messages_json = excluded.ui_messages_json,
                    handler_state_json = excluded.handler_state_json,
                    message_count = excluded.message_count
                ''',
                (
                    data['session_id'],
                    data.get('created_at') or datetime.now().isoformat(),
                    data.get('updated_at') or datetime.now().isoformat(),
                    data.get('title') or 'New Chat',
                    json.dumps(data.get('llm_history') or [], ensure_ascii=False, default=str),
                    json.dumps(ui_messages, ensure_ascii=False, default=str),
                    json.dumps(data.get('handler_state'), ensure_ascii=False, default=str),
                    len(ui_messages),
                ),
            )

    def _load_from_db(self, session_id):
        with self._connect() as conn:
            row = conn.execute(
                '''
                SELECT session_id, created_at, updated_at, title,
                       llm_history_json, ui_messages_json, handler_state_json
                FROM sessions
                WHERE session_id = ?
                ''',
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            handler_state = json.loads(row['handler_state_json'])
            return {
                'session_id': row['session_id'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'title': row['title'],
                'llm_history': json.loads(row['llm_history_json']),
                'ui_messages': json.loads(row['ui_messages_json']),
                'handler_state': handler_state,
            }
        except (TypeError, json.JSONDecodeError) as e:
            print(f'[Warn] session {session_id} corrupt in sqlite: {e}')
            return None
