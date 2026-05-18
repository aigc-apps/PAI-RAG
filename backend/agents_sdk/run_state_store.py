"""SQLite CRUD for ``agent_run_states`` (RunState blobs) + GC.

The table is created in :mod:`backend.session_store_sqlite` alongside the
existing ``sessions``/``runs``/``responses`` tables; this module owns the
read/write path. Resume uses ``RunState.from_string()`` which is async, so
all read methods that hydrate a SDK ``RunState`` are async; raw CRUD that
just returns the JSON blob is sync.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from backend.session_store_base import SERVER_USER_ID
from backend.agents_sdk.lifecycle import (
    RUN_STATE_ACTIVE,
    RUN_STATE_EXPIRED,
    RUN_STATE_REQUIRES_ACTION,
    RUN_STATE_RUNNING,
    RUN_STATE_TERMINAL,
)

DEFAULT_TTL_SECONDS = 7 * 24 * 3600
DEFAULT_COMPLETED_TTL_SECONDS = 24 * 3600
DEFAULT_GC_INTERVAL_SECONDS = 300


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_in(delta_seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_seconds)).isoformat()


class RunStateStore:
    """Owns the ``agent_run_states`` table. Connections come from the same
    SQLite file as :class:`backend.session_store_sqlite.SQLiteSessionStore`;
    callers pass a connection factory so tests can swap in an in-memory DB.
    """

    def __init__(self, connect):
        self._connect = connect

    def upsert(
        self,
        *,
        id: str,
        session_id: str,
        run_id: str,
        response_id: str | None,
        user_id: str,
        model: str,
        status: str,
        run_state_blob: str,
        pending_interruption_json: str | None,
        last_event_id: str | None,
        audit_log_id: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        now = _iso_now()
        expires = _iso_in(ttl_seconds)
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO agent_run_states (
                    id, session_id, run_id, response_id, user_id, model, status,
                    run_state_blob, pending_interruption_json, last_event_id,
                    audit_log_id, created_at, last_active_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    run_state_blob = excluded.run_state_blob,
                    pending_interruption_json = excluded.pending_interruption_json,
                    last_event_id = excluded.last_event_id,
                    last_active_at = excluded.last_active_at,
                    expires_at = excluded.expires_at,
                    model = excluded.model
                ''',
                (
                    id, session_id, run_id, response_id, user_id, model, status,
                    run_state_blob, pending_interruption_json, last_event_id,
                    audit_log_id, now, now, expires,
                ),
            )

    def get(self, id: str, *, user_id: str = SERVER_USER_ID) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT * FROM agent_run_states WHERE id = ?',
                (id,),
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        if data['user_id'] != user_id and user_id != SERVER_USER_ID:
            return None
        return data

    def get_by_response_id(self, response_id: str, *, user_id: str = SERVER_USER_ID) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT * FROM agent_run_states WHERE response_id = ?',
                (response_id,),
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        if data['user_id'] != user_id and user_id != SERVER_USER_ID:
            return None
        return data

    def get_by_session_run(self, session_id: str, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT * FROM agent_run_states WHERE session_id = ? AND run_id = ?',
                (session_id, run_id),
            ).fetchone()
        return dict(row) if row else None

    def find_by_pending_call_id(
        self, call_id: str, *, session_id: str | None = None,
        user_id: str = SERVER_USER_ID,
    ) -> dict | None:
        """Find the ``requires_action`` row whose pending interruption list
        contains ``call_id``. Used by the chat-completions wire to resolve a
        ``role:'tool'`` resume into the right RunState.

        Match is a substring of the JSON-encoded ``pending_interruption_json``
        column — coarse but cheap, and there's typically only one paused row
        per session at a time. Caller can scope by ``session_id`` to avoid
        cross-session matches in shared workspaces.
        """
        sql = (
            'SELECT * FROM agent_run_states '
            'WHERE status = ? AND pending_interruption_json LIKE ?'
        )
        args: list = [RUN_STATE_REQUIRES_ACTION, f'%"call_id": "{call_id}"%']
        if session_id is not None:
            sql += ' AND session_id = ?'
            args.append(session_id)
        sql += ' ORDER BY last_active_at DESC LIMIT 1'
        with self._connect() as conn:
            row = conn.execute(sql, args).fetchone()
        if not row:
            return None
        data = dict(row)
        if data['user_id'] != user_id and user_id != SERVER_USER_ID:
            return None
        return data

    def mark_status(self, id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                'UPDATE agent_run_states SET status = ?, last_active_at = ? WHERE id = ?',
                (status, _iso_now(), id),
            )

    def list_active(self, *, user_id: str | None = None, limit: int = 50) -> list[dict]:
        sql = 'SELECT * FROM agent_run_states WHERE status IN (?, ?)'
        args: list = [RUN_STATE_RUNNING, RUN_STATE_REQUIRES_ACTION]
        if user_id is not None:
            sql += ' AND user_id = ?'
            args.append(user_id)
        sql += ' ORDER BY last_active_at DESC LIMIT ?'
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def gc_expired(self, completed_ttl_seconds: int = DEFAULT_COMPLETED_TTL_SECONDS) -> int:
        """Sweep: soft-mark expired rows past expires_at; delete rows with
        terminal status older than ``completed_ttl_seconds``. Returns rows
        affected.
        """
        now = _iso_now()
        completed_cutoff = _iso_in(-completed_ttl_seconds)
        with self._connect() as conn:
            cur = conn.execute(
                'UPDATE agent_run_states SET status = ? WHERE status IN (?, ?) AND expires_at < ?',
                (RUN_STATE_EXPIRED, RUN_STATE_RUNNING, RUN_STATE_REQUIRES_ACTION, now),
            )
            soft = cur.rowcount or 0
            placeholders = ','.join('?' for _ in RUN_STATE_TERMINAL)
            cur = conn.execute(
                f'DELETE FROM agent_run_states WHERE status IN ({placeholders}) AND last_active_at < ?',
                (*RUN_STATE_TERMINAL, completed_cutoff),
            )
            hard = cur.rowcount or 0
        return soft + hard

    def delete(self, id: str) -> None:
        with self._connect() as conn:
            conn.execute('DELETE FROM agent_run_states WHERE id = ?', (id,))
