"""SQLite-backed storage for agent endpoints and judge model config.

Resolution is DB-first with env-var fallback:
  - `load_active_agent_record(slot)` returns the active agent row for slot
    'a' or 'b', or None if no row is bound.
  - `load_judge_record()` returns the singleton judge row, or None.

API keys are NEVER stored in the DB. Each record stores `api_key_env`, the
name of an environment variable; callers resolve the live key via
`os.getenv(api_key_env)` at request time.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

VALID_TRACE_MODES = {"responses", "chat", "runs", "openclaw"}

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        name TEXT NOT NULL,
        base_url TEXT NOT NULL,
        model TEXT NOT NULL,
        trace_mode TEXT NOT NULL,
        api_key_env TEXT NOT NULL DEFAULT '',
        runs_base_url TEXT NOT NULL DEFAULT '',
        headers_json TEXT NOT NULL DEFAULT '{}',
        description TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS active_pair (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        a_agent_id TEXT,
        b_agent_id TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (a_agent_id) REFERENCES agents(id),
        FOREIGN KEY (b_agent_id) REFERENCES agents(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS judge_config (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        base_url TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL DEFAULT '',
        api_key_env TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_agents_created_at ON agents(created_at DESC)",
)


def init_tables(conn: sqlite3.Connection) -> None:
    for stmt in DDL_STATEMENTS:
        conn.execute(stmt)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return f"agt_{uuid.uuid4().hex[:12]}"


def _row_to_agent_dict(row: sqlite3.Row) -> dict[str, Any]:
    headers_raw = row["headers_json"] or "{}"
    try:
        headers = json.loads(headers_raw)
        if not isinstance(headers, dict):
            headers = {}
    except json.JSONDecodeError:
        headers = {}
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "name": row["name"],
        "base_url": row["base_url"],
        "model": row["model"],
        "trace_mode": row["trace_mode"],
        "api_key_env": row["api_key_env"] or "",
        "runs_base_url": row["runs_base_url"] or "",
        "headers": headers,
        "description": row["description"] or "",
    }


def list_agents(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM agents ORDER BY created_at ASC"
    ).fetchall()
    return [_row_to_agent_dict(r) for r in rows]


def get_agent(conn: sqlite3.Connection, agent_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
    return _row_to_agent_dict(row) if row else None


def create_agent(
    conn: sqlite3.Connection,
    *,
    name: str,
    base_url: str,
    model: str,
    trace_mode: str = "responses",
    api_key_env: str = "",
    runs_base_url: str = "",
    headers: dict[str, str] | None = None,
    description: str = "",
) -> dict[str, Any]:
    if trace_mode not in VALID_TRACE_MODES:
        trace_mode = "responses"
    now = _now()
    agent_id = _new_id()
    conn.execute(
        """
        INSERT INTO agents (
            id, created_at, updated_at, name, base_url, model, trace_mode,
            api_key_env, runs_base_url, headers_json, description
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            agent_id,
            now,
            now,
            name.strip() or "Untitled Agent",
            base_url.strip(),
            model.strip(),
            trace_mode,
            api_key_env.strip(),
            runs_base_url.strip(),
            json.dumps(headers or {}, ensure_ascii=False),
            description.strip(),
        ),
    )
    conn.commit()
    return get_agent(conn, agent_id)  # type: ignore[return-value]


def update_agent(
    conn: sqlite3.Connection,
    agent_id: str,
    *,
    name: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    trace_mode: str | None = None,
    api_key_env: str | None = None,
    runs_base_url: str | None = None,
    headers: dict[str, str] | None = None,
    description: str | None = None,
) -> dict[str, Any] | None:
    existing = get_agent(conn, agent_id)
    if not existing:
        return None
    fields: dict[str, Any] = {}
    if name is not None:
        fields["name"] = name.strip() or existing["name"]
    if base_url is not None:
        fields["base_url"] = base_url.strip()
    if model is not None:
        fields["model"] = model.strip()
    if trace_mode is not None:
        fields["trace_mode"] = trace_mode if trace_mode in VALID_TRACE_MODES else existing["trace_mode"]
    if api_key_env is not None:
        fields["api_key_env"] = api_key_env.strip()
    if runs_base_url is not None:
        fields["runs_base_url"] = runs_base_url.strip()
    if headers is not None:
        fields["headers_json"] = json.dumps(headers, ensure_ascii=False)
    if description is not None:
        fields["description"] = description.strip()
    if not fields:
        return existing
    fields["updated_at"] = _now()
    sets = ", ".join(f"{col} = ?" for col in fields)
    params = list(fields.values()) + [agent_id]
    conn.execute(f"UPDATE agents SET {sets} WHERE id = ?", params)
    conn.commit()
    return get_agent(conn, agent_id)


def delete_agent(conn: sqlite3.Connection, agent_id: str) -> bool:
    cursor = conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
    # If this agent was bound to active_pair, clear those bindings so resolution
    # falls back to env vars cleanly.
    conn.execute(
        "UPDATE active_pair SET a_agent_id = NULL, updated_at = ? WHERE a_agent_id = ?",
        (_now(), agent_id),
    )
    conn.execute(
        "UPDATE active_pair SET b_agent_id = NULL, updated_at = ? WHERE b_agent_id = ?",
        (_now(), agent_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def get_active_pair(conn: sqlite3.Connection) -> dict[str, str | None]:
    row = conn.execute("SELECT * FROM active_pair WHERE id = 1").fetchone()
    if row is None:
        return {"a_agent_id": None, "b_agent_id": None, "updated_at": None}
    return {
        "a_agent_id": row["a_agent_id"],
        "b_agent_id": row["b_agent_id"],
        "updated_at": row["updated_at"],
    }


def set_active_pair(
    conn: sqlite3.Connection,
    *,
    a_agent_id: str | None,
    b_agent_id: str | None,
) -> dict[str, str | None]:
    now = _now()
    conn.execute(
        """
        INSERT INTO active_pair (id, a_agent_id, b_agent_id, updated_at)
        VALUES (1, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            a_agent_id = excluded.a_agent_id,
            b_agent_id = excluded.b_agent_id,
            updated_at = excluded.updated_at
        """,
        (a_agent_id, b_agent_id, now),
    )
    conn.commit()
    return get_active_pair(conn)


def load_active_agent_record(
    conn: sqlite3.Connection, slot: Literal["a", "b"]
) -> dict[str, Any] | None:
    pair = get_active_pair(conn)
    agent_id = pair["a_agent_id"] if slot == "a" else pair["b_agent_id"]
    if not agent_id:
        return None
    return get_agent(conn, agent_id)


def get_judge_record(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM judge_config WHERE id = 1").fetchone()
    if row is None:
        return None
    return {
        "base_url": row["base_url"] or "",
        "model": row["model"] or "",
        "api_key_env": row["api_key_env"] or "",
        "updated_at": row["updated_at"],
    }


def set_judge_record(
    conn: sqlite3.Connection,
    *,
    base_url: str,
    model: str,
    api_key_env: str,
) -> dict[str, Any]:
    now = _now()
    conn.execute(
        """
        INSERT INTO judge_config (id, base_url, model, api_key_env, updated_at)
        VALUES (1, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            base_url = excluded.base_url,
            model = excluded.model,
            api_key_env = excluded.api_key_env,
            updated_at = excluded.updated_at
        """,
        (base_url.strip(), model.strip(), api_key_env.strip(), now),
    )
    conn.commit()
    return get_judge_record(conn)  # type: ignore[return-value]


def env_var_status(name: str) -> dict[str, Any]:
    if not name:
        return {"name": "", "present": False, "preview": ""}
    raw = os.getenv(name, "")
    return {
        "name": name,
        "present": bool(raw),
        "preview": _mask_secret(raw),
    }


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}***{value[-4:]}"


def migrate_from_env(conn: sqlite3.Connection) -> dict[str, Any]:
    """Seed the agent library and judge_config from legacy env vars on first
    boot. Idempotent — runs only when the relevant tables are empty.

    Returns a small report dict useful for startup logging.
    """
    report: dict[str, Any] = {"seeded_agents": [], "seeded_judge": False}

    existing = conn.execute("SELECT COUNT(*) AS n FROM agents").fetchone()
    if (existing["n"] if existing else 0) == 0:
        seeded_ids: dict[str, str | None] = {"a": None, "b": None}
        for slot, prefix, fallback_name in (
            ("a", "AGENT_A", "Agent A"),
            ("b", "AGENT_B", "Agent B"),
        ):
            base_url = os.getenv(f"{prefix}_BASE_URL", "").strip()
            if not base_url:
                continue
            trace_mode = os.getenv(f"{prefix}_TRACE_MODE", "responses").strip().lower()
            if trace_mode not in VALID_TRACE_MODES:
                trace_mode = "responses"
            api_key_env_name = (
                f"{prefix}_API_KEY" if os.getenv(f"{prefix}_API_KEY", "").strip() else ""
            )
            rec = create_agent(
                conn,
                name=os.getenv(f"{prefix}_NAME", fallback_name).strip() or fallback_name,
                base_url=base_url,
                model=os.getenv(f"{prefix}_MODEL", "hermes-agent").strip() or "hermes-agent",
                trace_mode=trace_mode,
                api_key_env=api_key_env_name,
                runs_base_url=os.getenv(f"{prefix}_RUNS_BASE_URL", "").strip(),
                description="Imported from environment on first boot",
            )
            seeded_ids[slot] = rec["id"]
            report["seeded_agents"].append({"slot": slot, "id": rec["id"], "name": rec["name"]})
        if seeded_ids["a"] or seeded_ids["b"]:
            set_active_pair(conn, a_agent_id=seeded_ids["a"], b_agent_id=seeded_ids["b"])

    if get_judge_record(conn) is None:
        base_url = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1").strip()
        model = os.getenv("JUDGE_MODEL", "").strip()
        # Prefer the dedicated judge key var if it exists; otherwise fall back
        # to OPENAI_API_KEY so existing deployments continue to work.
        api_key_env_name = ""
        if os.getenv("JUDGE_OPENAI_API_KEY", "").strip():
            api_key_env_name = "JUDGE_OPENAI_API_KEY"
        elif os.getenv("OPENAI_API_KEY", "").strip():
            api_key_env_name = "OPENAI_API_KEY"
        if base_url or model or api_key_env_name:
            set_judge_record(
                conn,
                base_url=base_url,
                model=model,
                api_key_env=api_key_env_name,
            )
            report["seeded_judge"] = True

    return report
