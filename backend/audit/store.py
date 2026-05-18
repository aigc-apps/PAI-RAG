"""SQLite-backed append-only writer for ``audit_events``.

Categories (kept stable for downstream queries):

- ``llm_chunk`` — one model output chunk (token / delta).
- ``tool_call`` — tool was invoked by the model.
- ``tool_result`` — tool produced an output.
- ``hitl_pause`` — run paused on a ``ToolApprovalItem`` interruption.
- ``hitl_resume`` — pause resolved; carries the user's answer (redacted).
- ``hitl_auto_continue`` — HITL interruption was resolved by autonomous mode.
- ``agent_handoff`` — multi-agent handoff (currently unused but reserved).
- ``run_complete`` — terminal success.
- ``run_failed`` — terminal failure.
- ``error`` — non-terminal error.

Payloads are sanitized through :func:`agent_loop.redact_sensitive_text`
before they hit disk, matching the redaction the legacy archive path uses.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterable

from agent_loop import redact_sensitive_text

CATEGORY_LLM_CHUNK = 'llm_chunk'
CATEGORY_TOOL_CALL = 'tool_call'
CATEGORY_TOOL_RESULT = 'tool_result'
CATEGORY_HITL_PAUSE = 'hitl_pause'
CATEGORY_HITL_RESUME = 'hitl_resume'
CATEGORY_HITL_AUTO_CONTINUE = 'hitl_auto_continue'
CATEGORY_AGENT_HANDOFF = 'agent_handoff'
CATEGORY_RUN_COMPLETE = 'run_complete'
CATEGORY_RUN_FAILED = 'run_failed'
CATEGORY_ERROR = 'error'

ALL_CATEGORIES = frozenset({
    CATEGORY_LLM_CHUNK, CATEGORY_TOOL_CALL, CATEGORY_TOOL_RESULT,
    CATEGORY_HITL_PAUSE, CATEGORY_HITL_RESUME, CATEGORY_HITL_AUTO_CONTINUE,
    CATEGORY_AGENT_HANDOFF, CATEGORY_RUN_COMPLETE, CATEGORY_RUN_FAILED,
    CATEGORY_ERROR,
})


@dataclass
class AuditEvent:
    audit_log_id: str
    run_id: str
    session_id: str
    response_id: str | None
    category: str
    payload: dict
    ts_ms: int | None = None


def _sanitize(payload):
    """Walk a JSON-able payload and apply ``redact_sensitive_text`` to all
    string leaves. Lists/dicts are recursed in place. ``None``/numbers/bools
    pass through unchanged.
    """
    if isinstance(payload, str):
        return redact_sensitive_text(payload)
    if isinstance(payload, dict):
        return {k: _sanitize(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_sanitize(v) for v in payload]
    return payload


class AuditStore:
    """Synchronous SQLite writer. Pass a connection factory matching the one
    used by :class:`backend.session_store_sqlite.SQLiteSessionStore` so we
    write to the same database file.
    """

    def __init__(self, connect):
        self._connect = connect

    def append(self, event: AuditEvent) -> None:
        if event.category not in ALL_CATEGORIES:
            raise ValueError(f'unknown audit category: {event.category!r}')
        ts_ms = event.ts_ms if event.ts_ms is not None else int(time.time() * 1000)
        payload_json = json.dumps(_sanitize(event.payload), ensure_ascii=False, default=str)
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO audit_events (
                    audit_log_id, run_id, session_id, response_id, ts_ms, category, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    event.audit_log_id, event.run_id, event.session_id,
                    event.response_id, ts_ms, event.category, payload_json,
                ),
            )

    def append_many(self, events: Iterable[AuditEvent]) -> int:
        rows = []
        now = int(time.time() * 1000)
        for ev in events:
            if ev.category not in ALL_CATEGORIES:
                raise ValueError(f'unknown audit category: {ev.category!r}')
            rows.append((
                ev.audit_log_id, ev.run_id, ev.session_id, ev.response_id,
                ev.ts_ms if ev.ts_ms is not None else now,
                ev.category,
                json.dumps(_sanitize(ev.payload), ensure_ascii=False, default=str),
            ))
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                '''
                INSERT INTO audit_events (
                    audit_log_id, run_id, session_id, response_id, ts_ms, category, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                rows,
            )
        return len(rows)

    def query(self, audit_log_id: str, *, limit: int = 1000) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                '''
                SELECT id, audit_log_id, run_id, session_id, response_id, ts_ms, category, payload_json
                FROM audit_events
                WHERE audit_log_id = ?
                ORDER BY ts_ms ASC, id ASC
                LIMIT ?
                ''',
                (audit_log_id, limit),
            ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            try:
                data['payload'] = json.loads(data.pop('payload_json'))
            except (TypeError, ValueError):
                data['payload'] = None
            out.append(data)
        return out

    def purge_older_than(self, days: int) -> int:
        cutoff = int((time.time() - days * 86400) * 1000)
        with self._connect() as conn:
            cur = conn.execute('DELETE FROM audit_events WHERE ts_ms < ?', (cutoff,))
            return cur.rowcount or 0
