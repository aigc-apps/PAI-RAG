"""Compatibility shim: re-exports the SQLite-backed session store.

Existing call sites do `from session_store import SessionStore, SERVER_USER_ID, ...`.
The implementation lives in `backend.session_store_sqlite` (concrete) and
`backend.session_store_base` (abstract interface + backend-agnostic helpers).
A future migration may swap the default implementation here without touching
import sites; once all imports are migrated to the explicit modules, this
shim can be deleted.
"""
from backend.session_store_base import (
    ACTIVE_STATUSES,
    BaseSessionStore,
    DEFAULT_SESSION_TITLE,
    SERVER_USER_ID,
    session_title_from_messages,
)
from backend.session_store_sqlite import (
    SQLITE_JOURNAL_MODES,
    SQLiteSessionStore,
    sqlite_journal_mode,
)

# Public alias kept for backward compatibility with `from session_store import SessionStore`.
SessionStore = SQLiteSessionStore

__all__ = [
    'ACTIVE_STATUSES',
    'BaseSessionStore',
    'DEFAULT_SESSION_TITLE',
    'SERVER_USER_ID',
    'SQLITE_JOURNAL_MODES',
    'SQLiteSessionStore',
    'SessionStore',
    'session_title_from_messages',
    'sqlite_journal_mode',
]
