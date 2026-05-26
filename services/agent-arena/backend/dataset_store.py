"""SQLite-backed storage for evaluation datasets, cases, and eval runs.

Data model:
  - datasets        : named, isolated collections (e.g., "金融 FAQ v1")
  - dataset_cases   : individual evaluation cases with optional expected_answer
  - dataset_runs    : a single execution of a dataset against the active A/B pair
  - dataset_run_items : per-case rows under a run, holding A/B run_ids and judge result_id

Cases store an optional `source_run_id` linking back to an arena_runs row when
the case was harvested from a live run via the "加入评测集" flow.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_cases (
        id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        query TEXT NOT NULL,
        system_prompt TEXT NOT NULL DEFAULT '',
        expected_answer TEXT NOT NULL DEFAULT '',
        tags_json TEXT NOT NULL DEFAULT '[]',
        source_run_id TEXT,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_runs (
        id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        finished_at TEXT,
        status TEXT NOT NULL,
        agent_a_id TEXT,
        agent_b_id TEXT,
        judge_model TEXT,
        summary_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_run_items (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        case_id TEXT NOT NULL,
        idx INTEGER NOT NULL,
        status TEXT NOT NULL,
        a_run_id TEXT,
        b_run_id TEXT,
        judge_result_id TEXT,
        item_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY (run_id) REFERENCES dataset_runs(id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dataset_cases_dataset ON dataset_cases(dataset_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_dataset_runs_dataset ON dataset_runs(dataset_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_dataset_run_items_run ON dataset_run_items(run_id, idx ASC)",
)


def init_tables(conn: sqlite3.Connection) -> None:
    for stmt in DDL_STATEMENTS:
        conn.execute(stmt)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _row_to_dataset(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "name": row["name"],
        "description": row["description"] or "",
    }


def _row_to_case(row: sqlite3.Row) -> dict[str, Any]:
    raw = row["tags_json"] or "[]"
    try:
        tags = json.loads(raw)
        if not isinstance(tags, list):
            tags = []
    except json.JSONDecodeError:
        tags = []
    return {
        "id": row["id"],
        "dataset_id": row["dataset_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "query": row["query"],
        "system_prompt": row["system_prompt"] or "",
        "expected_answer": row["expected_answer"] or "",
        "tags": [str(t) for t in tags],
        "source_run_id": row["source_run_id"],
    }


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    summary_raw = row["summary_json"] or "{}"
    try:
        summary = json.loads(summary_raw)
        if not isinstance(summary, dict):
            summary = {}
    except json.JSONDecodeError:
        summary = {}
    return {
        "id": row["id"],
        "dataset_id": row["dataset_id"],
        "created_at": row["created_at"],
        "finished_at": row["finished_at"],
        "status": row["status"],
        "agent_a_id": row["agent_a_id"],
        "agent_b_id": row["agent_b_id"],
        "judge_model": row["judge_model"],
        "summary": summary,
    }


def _row_to_run_item(row: sqlite3.Row) -> dict[str, Any]:
    raw = row["item_json"] or "{}"
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            body = {}
    except json.JSONDecodeError:
        body = {}
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "case_id": row["case_id"],
        "idx": row["idx"],
        "status": row["status"],
        "a_run_id": row["a_run_id"],
        "b_run_id": row["b_run_id"],
        "judge_result_id": row["judge_result_id"],
        "body": body,
    }


# ---------- datasets ----------

def list_datasets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM datasets ORDER BY created_at DESC"
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        ds = _row_to_dataset(row)
        ds["case_count"] = conn.execute(
            "SELECT COUNT(*) AS n FROM dataset_cases WHERE dataset_id = ?",
            (row["id"],),
        ).fetchone()["n"]
        last_run = conn.execute(
            "SELECT * FROM dataset_runs WHERE dataset_id = ? ORDER BY created_at DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
        ds["last_run"] = _row_to_run(last_run) if last_run else None
        out.append(ds)
    return out


def get_dataset(conn: sqlite3.Connection, dataset_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    return _row_to_dataset(row) if row else None


def create_dataset(
    conn: sqlite3.Connection, *, name: str, description: str = ""
) -> dict[str, Any]:
    now = _now()
    dataset_id = _new_id("ds")
    conn.execute(
        "INSERT INTO datasets (id, created_at, updated_at, name, description) VALUES (?, ?, ?, ?, ?)",
        (dataset_id, now, now, name.strip() or "Untitled Dataset", description.strip()),
    )
    conn.commit()
    return get_dataset(conn, dataset_id)  # type: ignore[return-value]


def update_dataset(
    conn: sqlite3.Connection,
    dataset_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any] | None:
    existing = get_dataset(conn, dataset_id)
    if not existing:
        return None
    fields: dict[str, Any] = {}
    if name is not None:
        fields["name"] = name.strip() or existing["name"]
    if description is not None:
        fields["description"] = description.strip()
    if not fields:
        return existing
    fields["updated_at"] = _now()
    sets = ", ".join(f"{col} = ?" for col in fields)
    params = list(fields.values()) + [dataset_id]
    conn.execute(f"UPDATE datasets SET {sets} WHERE id = ?", params)
    conn.commit()
    return get_dataset(conn, dataset_id)


def delete_dataset(conn: sqlite3.Connection, dataset_id: str) -> bool:
    # Cascade by hand — SQLite FK enforcement is off by default in our connection.
    conn.execute(
        "DELETE FROM dataset_run_items WHERE run_id IN (SELECT id FROM dataset_runs WHERE dataset_id = ?)",
        (dataset_id,),
    )
    conn.execute("DELETE FROM dataset_runs WHERE dataset_id = ?", (dataset_id,))
    conn.execute("DELETE FROM dataset_cases WHERE dataset_id = ?", (dataset_id,))
    cursor = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    conn.commit()
    return cursor.rowcount > 0


# ---------- cases ----------

def list_cases(conn: sqlite3.Connection, dataset_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM dataset_cases WHERE dataset_id = ? ORDER BY created_at ASC",
        (dataset_id,),
    ).fetchall()
    return [_row_to_case(r) for r in rows]


def get_case(conn: sqlite3.Connection, case_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM dataset_cases WHERE id = ?", (case_id,)).fetchone()
    return _row_to_case(row) if row else None


def create_case(
    conn: sqlite3.Connection,
    *,
    dataset_id: str,
    query: str,
    system_prompt: str = "",
    expected_answer: str = "",
    tags: list[str] | None = None,
    source_run_id: str | None = None,
) -> dict[str, Any]:
    now = _now()
    case_id = _new_id("case")
    conn.execute(
        """
        INSERT INTO dataset_cases (
            id, dataset_id, created_at, updated_at, query,
            system_prompt, expected_answer, tags_json, source_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_id,
            dataset_id,
            now,
            now,
            query.strip(),
            system_prompt.strip(),
            expected_answer.strip(),
            json.dumps([str(t) for t in (tags or [])], ensure_ascii=False),
            source_run_id,
        ),
    )
    conn.commit()
    return get_case(conn, case_id)  # type: ignore[return-value]


def update_case(
    conn: sqlite3.Connection,
    case_id: str,
    *,
    query: str | None = None,
    system_prompt: str | None = None,
    expected_answer: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any] | None:
    existing = get_case(conn, case_id)
    if not existing:
        return None
    fields: dict[str, Any] = {}
    if query is not None:
        fields["query"] = query.strip()
    if system_prompt is not None:
        fields["system_prompt"] = system_prompt.strip()
    if expected_answer is not None:
        fields["expected_answer"] = expected_answer.strip()
    if tags is not None:
        fields["tags_json"] = json.dumps([str(t) for t in tags], ensure_ascii=False)
    if not fields:
        return existing
    fields["updated_at"] = _now()
    sets = ", ".join(f"{col} = ?" for col in fields)
    params = list(fields.values()) + [case_id]
    conn.execute(f"UPDATE dataset_cases SET {sets} WHERE id = ?", params)
    conn.commit()
    return get_case(conn, case_id)


def delete_case(conn: sqlite3.Connection, case_id: str) -> bool:
    cursor = conn.execute("DELETE FROM dataset_cases WHERE id = ?", (case_id,))
    conn.commit()
    return cursor.rowcount > 0


# ---------- runs ----------

def create_run(
    conn: sqlite3.Connection,
    *,
    dataset_id: str,
    agent_a_id: str | None,
    agent_b_id: str | None,
    judge_model: str | None,
) -> dict[str, Any]:
    run_id = _new_id("dsrun")
    now = _now()
    conn.execute(
        """
        INSERT INTO dataset_runs (
            id, dataset_id, created_at, finished_at, status,
            agent_a_id, agent_b_id, judge_model, summary_json
        ) VALUES (?, ?, ?, NULL, 'running', ?, ?, ?, '{}')
        """,
        (run_id, dataset_id, now, agent_a_id, agent_b_id, judge_model),
    )
    conn.commit()
    return get_run(conn, run_id)  # type: ignore[return-value]


def get_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM dataset_runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_run(row) if row else None


def list_runs(conn: sqlite3.Connection, dataset_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM dataset_runs WHERE dataset_id = ? ORDER BY created_at DESC",
        (dataset_id,),
    ).fetchall()
    return [_row_to_run(r) for r in rows]


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: str,
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    conn.execute(
        "UPDATE dataset_runs SET status = ?, finished_at = ?, summary_json = ? WHERE id = ?",
        (status, _now(), json.dumps(summary, ensure_ascii=False), run_id),
    )
    conn.commit()
    return get_run(conn, run_id)


def upsert_run_item(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    case_id: str,
    idx: int,
    status: str,
    a_run_id: str | None = None,
    b_run_id: str | None = None,
    judge_result_id: str | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item_id = _new_id("dsri")
    conn.execute(
        """
        INSERT INTO dataset_run_items (
            id, run_id, case_id, idx, status,
            a_run_id, b_run_id, judge_result_id, item_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item_id,
            run_id,
            case_id,
            idx,
            status,
            a_run_id,
            b_run_id,
            judge_result_id,
            json.dumps(body or {}, ensure_ascii=False),
        ),
    )
    conn.commit()
    return _row_to_run_item(
        conn.execute("SELECT * FROM dataset_run_items WHERE id = ?", (item_id,)).fetchone()
    )


def list_run_items(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM dataset_run_items WHERE run_id = ? ORDER BY idx ASC",
        (run_id,),
    ).fetchall()
    return [_row_to_run_item(r) for r in rows]
