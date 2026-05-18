"""Backend-agnostic session-store interface.

`BaseSessionStore` declares the public contract that the rest of the codebase
talks to (`backend.agent_service`, the SDK runner, tests). The current concrete
implementation is `backend.session_store_sqlite.SQLiteSessionStore`; future
Postgres / Tablestore backends should subclass `BaseSessionStore` so call
sites stay identical.

Constants and the `session_title_from_messages` helper live here too because
they are part of the public contract — independent of the storage engine.
"""
from abc import ABC, abstractmethod


SERVER_USER_ID = '__server__'
ACTIVE_STATUSES = {'running', 'waiting_user'}
DEFAULT_SESSION_TITLE = 'New Task'


def session_title_from_messages(ui_messages, current_title=None):
    title = (current_title or '').strip()
    if title and title != DEFAULT_SESSION_TITLE:
        return title
    return next(
        (
            (m.get('content') or '').strip()[:60]
            for m in ui_messages
            if m.get('role') == 'user' and (m.get('content') or '').strip()
        ),
        DEFAULT_SESSION_TITLE,
    )


class BaseSessionStore(ABC):
    """Abstract storage interface for sessions, runs, and Responses-API records."""

    @property
    @abstractmethod
    def db_path(self):
        ...

    # ----- sessions -----
    @abstractmethod
    def save(self, session_id, llm_history, ui_messages, handler_state=None,
             title=None, user_id=SERVER_USER_ID, status='idle',
             active_run_id=None, workspace_path=''):
        ...

    @abstractmethod
    def load(self, session_id, user_id=SERVER_USER_ID):
        ...

    @abstractmethod
    def list_sessions(self, user_id=SERVER_USER_ID):
        ...

    @abstractmethod
    def delete(self, session_id, user_id=SERVER_USER_ID):
        ...

    @abstractmethod
    def owner_for(self, session_id):
        ...

    @abstractmethod
    def session_exists(self, session_id):
        ...

    # ----- runs -----
    @abstractmethod
    def create_run_record(self, session_id, user_id, run_id, mode,
                          status='queued', metadata=None):
        ...

    @abstractmethod
    def try_start_run(self, session_id, user_id, run_id, mode, user_text,
                      workspace_path='', max_global_runs=0, max_user_runs=0):
        ...

    @abstractmethod
    def try_start_regenerate_run(self, session_id, user_id, run_id, mode,
                                 workspace_path='', max_global_runs=0,
                                 max_user_runs=0):
        ...

    @abstractmethod
    def answer_waiting_run(self, session_id, user_id, run_id, answer):
        ...

    @abstractmethod
    def mark_waiting_user(self, session_id, user_id, run_id):
        ...

    @abstractmethod
    def set_run_status(self, run_id, user_id, status):
        ...

    @abstractmethod
    def load_run(self, run_id, user_id=SERVER_USER_ID):
        ...

    @abstractmethod
    def set_run_last_event(self, run_id, user_id, event_id):
        ...

    @abstractmethod
    def finish_run(self, session_id, user_id, run_id, status, error=''):
        ...

    @abstractmethod
    def save_run_snapshot(self, session_id, user_id, run_id, llm_history,
                          ui_messages, handler_state=None, status='running',
                          active_run_id=None, workspace_path=''):
        ...

    @abstractmethod
    def request_cancel(self, session_id, user_id, run_id):
        ...

    # ----- Responses API -----
    @abstractmethod
    def save_response(self, response_id, response, conversation_history=None,
                      instructions=None, session_id='',
                      user_id=SERVER_USER_ID, conversation=''):
        ...

    @abstractmethod
    def load_response(self, response_id, user_id=SERVER_USER_ID):
        ...

    @abstractmethod
    def delete_response(self, response_id, user_id=SERVER_USER_ID):
        ...

    @abstractmethod
    def latest_response_for_conversation(self, conversation,
                                         user_id=SERVER_USER_ID):
        ...
