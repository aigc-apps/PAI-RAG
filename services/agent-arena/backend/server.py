from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import sqlite3
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from . import config_store, dataset_store


ROOT_DIR = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
LOG_DIR = ROOT_DIR / "logs"
FRONTEND_LOG_PATH = LOG_DIR / "frontend.log"
DATA_DIR = ROOT_DIR / "data"
HISTORY_DB_PATH = Path(os.getenv("HISTORY_DB_PATH") or (DATA_DIR / "arena_history.sqlite3"))
MAX_TRACE_EVENTS = 2000
MAX_EVENT_TEXT = 1800
SUPPRESSED_TRACE_EVENTS = {"tool.delta", "tool.updated", "tool_call_delta"}
# Tool names whose function_call_arguments stream is redundant with the final
# user-visible output. The args carry the same content already accumulated via
# response.output_text — recording them just bloats the trace.
SUPPRESSED_ARG_TOOLS = {"final_report"}

frontend_logger = logging.getLogger("agent_arena.frontend")
frontend_logger.setLevel(logging.INFO)
frontend_logger.propagate = False
ACTIVE_FRONTEND_LOG_PATH = FRONTEND_LOG_PATH
if not frontend_logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    frontend_logger.addHandler(stream_handler)
    fallback_dir = Path("/tmp/agent_arena_logs")
    try:
        LOG_DIR.mkdir(exist_ok=True)
        log_path = FRONTEND_LOG_PATH
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError:
        fallback_dir.mkdir(exist_ok=True)
        log_path = fallback_dir / "frontend.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    ACTIVE_FRONTEND_LOG_PATH = log_path
    file_handler.setFormatter(formatter)
    frontend_logger.addHandler(file_handler)


@dataclass(frozen=True)
class AgentConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    trace_mode: str
    runs_base_url: str

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    @property
    def completion_url(self) -> str:
        return normalize_completion_url(self.base_url)

    @property
    def responses_url(self) -> str:
        return normalize_responses_url(self.base_url)

    @property
    def runs_url(self) -> str:
        return normalize_runs_url(self.runs_base_url or self.base_url)


@dataclass(frozen=True)
class JudgeConfig:
    base_url: str
    api_key: str
    model: str

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    @property
    def completion_url(self) -> str:
        return normalize_completion_url(self.base_url)


class CompareRequest(BaseModel):
    input: str = Field(..., min_length=1)
    system: str = ""
    temperature: float = Field(0.2, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=1, le=200000)


class AgentTraceEvent(BaseModel):
    event: str
    timestamp: float | None = None
    tool: str | None = None
    preview: str | None = None
    duration: float | None = None
    error: bool | str | None = None
    text: str | None = None
    delta: str | None = None
    # Responses API output_item type (e.g. "function_call", "message",
    # "reasoning"). Lets summarize_trace count function_call items as tool
    # invocations even though the event name is response.output_item.added
    # rather than the agents-trace native tool.started.
    item_type: str | None = None


class AgentResult(BaseModel):
    ok: bool
    name: str
    model: str
    content: str = ""
    latency_ms: int | None = None
    error: str | None = None
    raw_finish_reason: str | None = None
    trace_supported: bool = False
    trace_events: list[AgentTraceEvent] = Field(default_factory=list)
    trace_summary: dict[str, Any] = Field(default_factory=dict)


class CompareResponse(BaseModel):
    run_id: str
    request: dict[str, Any]
    agents: dict[str, AgentResult]


class JudgeRequest(BaseModel):
    run_id: str | None = None
    input: str = Field(..., min_length=1)
    system: str = ""
    agent_a: AgentResult
    agent_b: AgentResult


class JudgeResponse(BaseModel):
    ok: bool
    run_id: str | None = None
    judge_id: str | None = None
    winner: str | None = None
    summary: str = ""
    answer_scores: dict[str, Any] = Field(default_factory=dict)
    process_scores: dict[str, Any] = Field(default_factory=dict)
    strengths: dict[str, list[str]] = Field(default_factory=dict)
    weaknesses: dict[str, list[str]] = Field(default_factory=dict)
    recommendations: dict[str, list[str]] = Field(default_factory=dict)
    latency_ms: int | None = None
    error: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class FrontendLogRequest(BaseModel):
    level: str = "error"
    message: str = Field(..., min_length=1)
    stack: str | None = None
    component_stack: str | None = None
    source: str = "frontend"
    url: str | None = None
    user_agent: str | None = None
    timestamp: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class HistoryJudgeRecord(BaseModel):
    judge_id: str
    run_id: str
    created_at: str
    model: str
    result: JudgeResponse


class HistorySummary(BaseModel):
    kind: Literal["arena"] = "arena"
    run_id: str
    created_at: str
    updated_at: str
    input: str
    system: str
    temperature: float | None = None
    max_tokens: int | None = None
    agent_a: dict[str, Any]
    agent_b: dict[str, Any]
    judge_count: int = 0
    latest_judge: HistoryJudgeRecord | None = None


class BatchHistorySummary(BaseModel):
    kind: Literal["batch"] = "batch"
    batch_id: str
    created_at: str
    target: str
    mode: str
    iterations: int
    concurrency: int
    cancelled: bool = False
    input: str = ""
    item_count: int = 0
    success_count: int = 0
    success_rate: float | None = None
    agent_summaries: list[dict[str, Any]] = Field(default_factory=list)
    consistency_count: int = 0
    latest_consistency: dict[str, Any] | None = None


HistoryItem = Annotated[
    Union[HistorySummary, BatchHistorySummary],
    Field(discriminator="kind"),
]


class HistoryListResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    limit: int
    offset: int


class HistoryDetailResponse(BaseModel):
    run_id: str
    created_at: str
    updated_at: str
    input: str
    system: str
    temperature: float | None = None
    max_tokens: int | None = None
    compare: CompareResponse
    judges: list[HistoryJudgeRecord] = Field(default_factory=list)


BatchTarget = Literal["a", "b", "both"]
BatchMode = Literal["form", "raw"]
AssertionKind = Literal["substring", "regex"]
BATCH_MAX_ITERATIONS = 200
BATCH_MAX_CONCURRENCY = 8


class BatchAssertion(BaseModel):
    type: AssertionKind = "substring"
    value: str = Field(..., min_length=1)
    case_sensitive: bool = False


class BatchFormPayload(BaseModel):
    input: str = Field(..., min_length=1)
    system: str = ""
    temperature: float = Field(0.2, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=1, le=200000)


class BatchRunRequest(BaseModel):
    target: BatchTarget = "a"
    mode: BatchMode = "form"
    iterations: int = Field(10, ge=1, le=BATCH_MAX_ITERATIONS)
    concurrency: int = Field(1, ge=1, le=BATCH_MAX_CONCURRENCY)
    form: BatchFormPayload | None = None
    raw_body: dict[str, Any] | None = None
    assertion: BatchAssertion | None = None

    @model_validator(mode="after")
    def _check_payload(self) -> "BatchRunRequest":
        if self.mode == "form" and self.form is None:
            raise ValueError("form payload required when mode='form'")
        if self.mode == "raw" and not self.raw_body:
            raise ValueError("raw_body required when mode='raw'")
        if self.assertion is not None and self.assertion.type == "regex":
            try:
                re.compile(self.assertion.value)
            except re.error as exc:
                raise ValueError(f"invalid regex: {exc}") from exc
        return self


class BatchRunItem(BaseModel):
    index: int
    agent_key: Literal["a", "b"]
    agent_name: str
    agent_model: str
    ok: bool
    latency_ms: int | None = None
    content: str = ""
    content_length: int = 0
    finish_reason: str | None = None
    error: str | None = None
    assertion_passed: bool | None = None
    trace_summary: dict[str, Any] = Field(default_factory=dict)
    trace_events: list[AgentTraceEvent] = Field(default_factory=list)


class BatchAgentSummary(BaseModel):
    agent_key: Literal["a", "b"]
    agent_name: str
    agent_model: str
    total: int
    success: int
    success_rate: float
    assertion_total: int = 0
    assertion_passed: int = 0
    assertion_rate: float | None = None
    latency_min_ms: int | None = None
    latency_p50_ms: int | None = None
    latency_p90_ms: int | None = None
    latency_max_ms: int | None = None
    content_len_min: int | None = None
    content_len_avg: float | None = None
    content_len_max: int | None = None
    tool_call_avg: float | None = None
    failed_tool_total: int = 0
    finish_reasons: dict[str, int] = Field(default_factory=dict)


class BatchRunResponse(BaseModel):
    batch_id: str
    created_at: str
    request: BatchRunRequest
    items: list[BatchRunItem]
    summaries: dict[str, BatchAgentSummary]
    cancelled: bool = False


class ConsistencyIssue(BaseModel):
    index: int
    agent_key: Literal["a", "b"]
    problem: str
    severity: Literal["low", "medium", "high"] = "medium"


class ConsistencyAgentReport(BaseModel):
    agent_key: Literal["a", "b"]
    agent_name: str
    samples_evaluated: int
    stable: bool
    consistency_score: int
    summary: str
    issues: list[ConsistencyIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class ConsistencyResponse(BaseModel):
    ok: bool
    batch_id: str
    created_at: str
    model: str = ""
    latency_ms: int | None = None
    reports: list[ConsistencyAgentReport] = Field(default_factory=list)
    error: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


# ---- Config & dataset payloads (DB-backed config; api_key_env references the
# name of an env var that holds the real secret) -----------------------------


class AgentDefCreate(BaseModel):
    name: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    trace_mode: Literal["responses", "chat", "runs", "openclaw"] = "responses"
    api_key_env: str = ""
    runs_base_url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    description: str = ""


class AgentDefPatch(BaseModel):
    name: str | None = None
    base_url: str | None = None
    model: str | None = None
    trace_mode: Literal["responses", "chat", "runs", "openclaw"] | None = None
    api_key_env: str | None = None
    runs_base_url: str | None = None
    headers: dict[str, str] | None = None
    description: str | None = None


class ActivePairSet(BaseModel):
    a_agent_id: str | None = None
    b_agent_id: str | None = None


class JudgeConfigSet(BaseModel):
    base_url: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    api_key_env: str = ""


class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""


class DatasetPatch(BaseModel):
    name: str | None = None
    description: str | None = None


class DatasetCaseCreate(BaseModel):
    query: str = Field(..., min_length=1)
    system_prompt: str = ""
    expected_answer: str = ""
    tags: list[str] = Field(default_factory=list)
    source_run_id: str | None = None


class DatasetCasePatch(BaseModel):
    query: str | None = None
    system_prompt: str | None = None
    expected_answer: str | None = None
    tags: list[str] | None = None


class HarvestCasePayload(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    expected_answer: str = ""
    query_override: str | None = None
    system_override: str | None = None
    tags: list[str] = Field(default_factory=list)


class BatchItemHarvestPayload(BaseModel):
    batch_id: str = Field(..., min_length=1)
    idx: int = Field(..., ge=0)
    agent_key: str = Field(..., min_length=1)  # "a" or "b"
    dataset_id: str = Field(..., min_length=1)
    expected_answer: str = ""
    query_override: str | None = None
    system_override: str | None = None
    tags: list[str] = Field(default_factory=list)


class DatasetRunStart(BaseModel):
    case_ids: list[str] | None = None  # None = all cases
    judge_each: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def model_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def json_dumps(value: Any) -> str:
    return json.dumps(model_to_dict(value), ensure_ascii=False, default=str)


def json_loads_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def init_history_db() -> None:
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS arena_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                input TEXT NOT NULL,
                system TEXT NOT NULL DEFAULT '',
                temperature REAL,
                max_tokens INTEGER,
                agent_a_name TEXT,
                agent_a_model TEXT,
                agent_a_ok INTEGER,
                agent_a_latency_ms INTEGER,
                agent_b_name TEXT,
                agent_b_model TEXT,
                agent_b_ok INTEGER,
                agent_b_latency_ms INTEGER,
                compare_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS judge_results (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                model TEXT,
                ok INTEGER,
                winner TEXT,
                summary TEXT,
                latency_ms INTEGER,
                error TEXT,
                judge_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES arena_runs(id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_arena_runs_created_at ON arena_runs(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_judge_results_run_id ON judge_results(run_id, created_at DESC)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                target TEXT NOT NULL,
                mode TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                concurrency INTEGER NOT NULL,
                cancelled INTEGER NOT NULL DEFAULT 0,
                request_json TEXT NOT NULL,
                summaries_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_run_items (
                batch_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                agent_key TEXT NOT NULL,
                ok INTEGER NOT NULL,
                latency_ms INTEGER,
                item_json TEXT NOT NULL,
                PRIMARY KEY (batch_id, idx, agent_key),
                FOREIGN KEY (batch_id) REFERENCES batch_runs(id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_batch_runs_created_at ON batch_runs(created_at DESC)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_consistency_results (
                id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                model TEXT,
                ok INTEGER NOT NULL,
                latency_ms INTEGER,
                error TEXT,
                result_json TEXT NOT NULL,
                FOREIGN KEY (batch_id) REFERENCES batch_runs(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_batch_consistency_batch_id "
            "ON batch_consistency_results(batch_id, created_at DESC)"
        )
        config_store.init_tables(conn)
        dataset_store.init_tables(conn)


def history_connect() -> sqlite3.Connection:
    init_history_db()
    conn = sqlite3.connect(HISTORY_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def agent_history_summary(result: AgentResult | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(result, AgentResult):
        data = model_to_dict(result)
    elif isinstance(result, dict):
        data = result
    else:
        data = {}
    return {
        "name": data.get("name") or "",
        "model": data.get("model") or "",
        "ok": bool(data.get("ok")),
        "latency_ms": data.get("latency_ms"),
        "error": data.get("error"),
        "content_preview": truncate_text(data.get("content") or "", 240),
        "trace_summary": data.get("trace_summary") or {},
    }


def save_compare_history(
    response: CompareResponse,
    user_input: str,
    system: str,
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    agents = response.agents
    result_a = agents["a"]
    result_b = agents["b"]
    now = utc_now()
    with history_connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO arena_runs (
                id, created_at, updated_at, input, system, temperature, max_tokens,
                agent_a_name, agent_a_model, agent_a_ok, agent_a_latency_ms,
                agent_b_name, agent_b_model, agent_b_ok, agent_b_latency_ms,
                compare_json
            )
            VALUES (?, COALESCE((SELECT created_at FROM arena_runs WHERE id = ?), ?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response.run_id,
                response.run_id,
                now,
                now,
                user_input,
                system,
                temperature,
                max_tokens,
                result_a.name,
                result_a.model,
                int(result_a.ok),
                result_a.latency_ms,
                result_b.name,
                result_b.model,
                int(result_b.ok),
                result_b.latency_ms,
                json_dumps(response),
            ),
        )
    return response.run_id


def save_compare_snapshot_from_judge_request(request: JudgeRequest, run_id: str) -> str:
    messages: list[dict[str, str]] = []
    if request.system.strip():
        messages.append({"role": "system", "content": request.system.strip()})
    messages.append({"role": "user", "content": request.input.strip()})
    compare = CompareResponse(
        run_id=run_id,
        request={"messages": messages, "temperature": None, "max_tokens": None},
        agents={"a": request.agent_a, "b": request.agent_b},
    )
    return save_compare_history(compare, request.input.strip(), request.system.strip(), None, None)


def history_run_exists(run_id: str) -> bool:
    with history_connect() as conn:
        row = conn.execute("SELECT 1 FROM arena_runs WHERE id = ?", (run_id,)).fetchone()
    return row is not None


def save_judge_history(run_id: str, response: JudgeResponse) -> str:
    judge_id = response.judge_id or new_id("judge")
    response.run_id = run_id
    response.judge_id = judge_id
    judge = get_judge_config()
    now = utc_now()
    with history_connect() as conn:
        conn.execute(
            """
            INSERT INTO judge_results (
                id, run_id, created_at, model, ok, winner, summary, latency_ms, error, judge_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                judge_id,
                run_id,
                now,
                judge.model,
                int(response.ok),
                response.winner,
                response.summary,
                response.latency_ms,
                response.error,
                json_dumps(response),
            ),
        )
        conn.execute("UPDATE arena_runs SET updated_at = ? WHERE id = ?", (now, run_id))
    return judge_id


def row_to_judge_record(row: sqlite3.Row | None) -> HistoryJudgeRecord | None:
    if row is None:
        return None
    data = json_loads_dict(row["judge_json"])
    if "run_id" not in data:
        data["run_id"] = row["run_id"]
    if "judge_id" not in data:
        data["judge_id"] = row["id"]
    return HistoryJudgeRecord(
        judge_id=row["id"],
        run_id=row["run_id"],
        created_at=row["created_at"],
        model=row["model"] or "",
        result=JudgeResponse(**data),
    )


def row_to_history_summary(row: sqlite3.Row) -> HistorySummary:
    compare_data = json_loads_dict(row["compare_json"])
    agents = compare_data.get("agents") if isinstance(compare_data.get("agents"), dict) else {}
    latest_judge = None
    if row["judge_id"]:
        data = json_loads_dict(row["judge_json"])
        data["run_id"] = data.get("run_id") or row["id"]
        data["judge_id"] = data.get("judge_id") or row["judge_id"]
        latest_judge = HistoryJudgeRecord(
            judge_id=row["judge_id"],
            run_id=row["id"],
            created_at=row["judge_created_at"],
            model=row["judge_model"] or "",
            result=JudgeResponse(**data),
        )
    return HistorySummary(
        run_id=row["id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        input=row["input"],
        system=row["system"] or "",
        temperature=row["temperature"],
        max_tokens=row["max_tokens"],
        agent_a=agent_history_summary(agents.get("a")),
        agent_b=agent_history_summary(agents.get("b")),
        judge_count=int(row["judge_count"] or 0),
        latest_judge=latest_judge,
    )


def row_to_batch_summary(row: sqlite3.Row) -> BatchHistorySummary:
    request_data = json_loads_dict(row["request_json"])
    summaries_data = json_loads_dict(row["summaries_json"])
    form = request_data.get("form") if isinstance(request_data.get("form"), dict) else {}
    raw_body = request_data.get("raw_body") if isinstance(request_data.get("raw_body"), dict) else {}
    user_input = str((form or {}).get("input") or (raw_body or {}).get("input") or "").strip()

    agent_summaries: list[dict[str, Any]] = []
    total_items = 0
    total_success = 0
    for key in ("a", "b"):
        s = summaries_data.get(key)
        if not isinstance(s, dict):
            continue
        total = int(s.get("total") or 0)
        success = int(s.get("success") or 0)
        total_items += total
        total_success += success
        agent_summaries.append(
            {
                "agent_key": key,
                "agent_name": s.get("agent_name") or f"Agent {key.upper()}",
                "agent_model": s.get("agent_model") or "",
                "total": total,
                "success": success,
                "success_rate": s.get("success_rate"),
                "latency_p50_ms": s.get("latency_p50_ms"),
                "latency_p90_ms": s.get("latency_p90_ms"),
            }
        )
    success_rate = round(total_success / total_items, 4) if total_items else None
    return BatchHistorySummary(
        batch_id=row["id"],
        created_at=row["created_at"],
        target=row["target"],
        mode=row["mode"],
        iterations=int(row["iterations"]),
        concurrency=int(row["concurrency"]),
        cancelled=bool(row["cancelled"]),
        input=user_input,
        item_count=total_items,
        success_count=total_success,
        success_rate=success_rate,
        agent_summaries=agent_summaries,
        consistency_count=int(row["consistency_count"] or 0),
        latest_consistency=(
            json.loads(row["latest_consistency_json"])
            if row["latest_consistency_json"]
            else None
        ),
    )


def list_history(limit: int, offset: int) -> HistoryListResponse:
    safe_limit = max(1, min(limit, 100))
    safe_offset = max(0, offset)
    with history_connect() as conn:
        arena_total = int(conn.execute("SELECT COUNT(*) FROM arena_runs").fetchone()[0])
        batch_total = int(conn.execute("SELECT COUNT(*) FROM batch_runs").fetchone()[0])
        total = arena_total + batch_total

        fetch_limit = safe_limit + safe_offset

        arena_rows = conn.execute(
            """
            SELECT
                r.*,
                COALESCE((SELECT COUNT(*) FROM judge_results j WHERE j.run_id = r.id), 0) AS judge_count,
                latest.id AS judge_id,
                latest.run_id AS judge_run_id,
                latest.created_at AS judge_created_at,
                latest.model AS judge_model,
                latest.judge_json AS judge_json
            FROM arena_runs r
            LEFT JOIN judge_results latest
                ON latest.id = (
                    SELECT j2.id
                    FROM judge_results j2
                    WHERE j2.run_id = r.id
                    ORDER BY j2.created_at DESC
                    LIMIT 1
                )
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            (fetch_limit,),
        ).fetchall()

        batch_rows = conn.execute(
            """
            SELECT
                b.*,
                COALESCE((SELECT COUNT(*) FROM batch_consistency_results c WHERE c.batch_id = b.id), 0) AS consistency_count,
                (
                    SELECT result_json
                    FROM batch_consistency_results c2
                    WHERE c2.batch_id = b.id
                    ORDER BY c2.created_at DESC
                    LIMIT 1
                ) AS latest_consistency_json
            FROM batch_runs b
            ORDER BY b.created_at DESC
            LIMIT ?
            """,
            (fetch_limit,),
        ).fetchall()

    items: list[HistoryItem] = []
    items.extend(row_to_history_summary(r) for r in arena_rows)
    items.extend(row_to_batch_summary(r) for r in batch_rows)
    items.sort(key=lambda x: x.created_at, reverse=True)
    paged = items[safe_offset : safe_offset + safe_limit]
    return HistoryListResponse(
        items=paged,
        total=total,
        limit=safe_limit,
        offset=safe_offset,
    )


def get_history_detail(run_id: str) -> HistoryDetailResponse | None:
    with history_connect() as conn:
        run = conn.execute("SELECT * FROM arena_runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            return None
        judge_rows = conn.execute(
            "SELECT * FROM judge_results WHERE run_id = ? ORDER BY created_at DESC",
            (run_id,),
        ).fetchall()
    compare_data = json_loads_dict(run["compare_json"])
    compare_data["run_id"] = compare_data.get("run_id") or run["id"]
    return HistoryDetailResponse(
        run_id=run["id"],
        created_at=run["created_at"],
        updated_at=run["updated_at"],
        input=run["input"],
        system=run["system"] or "",
        temperature=run["temperature"],
        max_tokens=run["max_tokens"],
        compare=CompareResponse(**compare_data),
        judges=[record for row in judge_rows if (record := row_to_judge_record(row)) is not None],
    )


def normalize_completion_url(base_url: str) -> str:
    clean = (base_url or "").strip().rstrip("/")
    if not clean:
        return ""
    if clean.endswith("/v1/chat/completions"):
        return clean
    if clean.endswith("/v1/responses"):
        return clean.rsplit("/responses", 1)[0] + "/chat/completions"
    if clean.endswith("/v1/runs"):
        return clean.rsplit("/runs", 1)[0] + "/chat/completions"
    if clean.endswith("/v1"):
        return f"{clean}/chat/completions"
    return f"{clean}/v1/chat/completions"


def normalize_responses_url(base_url: str) -> str:
    clean = (base_url or "").strip().rstrip("/")
    if not clean:
        return ""
    if clean.endswith("/v1/responses"):
        return clean
    if clean.endswith("/v1/chat/completions"):
        return clean.rsplit("/chat/completions", 1)[0] + "/responses"
    if clean.endswith("/v1/runs"):
        return clean.rsplit("/runs", 1)[0] + "/responses"
    if clean.endswith("/v1"):
        return f"{clean}/responses"
    return f"{clean}/v1/responses"


def normalize_runs_url(base_url: str) -> str:
    clean = (base_url or "").strip().rstrip("/")
    if not clean:
        return ""
    if clean.endswith("/v1/runs"):
        return clean
    if clean.endswith("/v1/chat/completions"):
        return clean.rsplit("/chat/completions", 1)[0] + "/runs"
    if clean.endswith("/v1/responses"):
        return clean.rsplit("/responses", 1)[0] + "/runs"
    if clean.endswith("/v1"):
        return f"{clean}/runs"
    return f"{clean}/v1/runs"


def _env_agent_config(prefix: str, fallback_name: str) -> AgentConfig:
    trace_mode = os.getenv(f"{prefix}_TRACE_MODE", "responses").strip().lower()
    if trace_mode not in {"responses", "chat", "runs", "openclaw"}:
        trace_mode = "responses"
    return AgentConfig(
        name=os.getenv(f"{prefix}_NAME", fallback_name).strip() or fallback_name,
        base_url=os.getenv(f"{prefix}_BASE_URL", "").strip(),
        api_key=os.getenv(f"{prefix}_API_KEY", "").strip(),
        model=os.getenv(f"{prefix}_MODEL", "hermes-agent").strip() or "hermes-agent",
        trace_mode=trace_mode,
        runs_base_url=os.getenv(f"{prefix}_RUNS_BASE_URL", "").strip(),
    )


def _record_to_agent_config(record: dict[str, Any]) -> AgentConfig:
    trace_mode = record.get("trace_mode", "responses")
    if trace_mode not in {"responses", "chat", "runs", "openclaw"}:
        trace_mode = "responses"
    api_key_env = record.get("api_key_env") or ""
    api_key = os.getenv(api_key_env, "").strip() if api_key_env else ""
    return AgentConfig(
        name=record.get("name") or "",
        base_url=record.get("base_url") or "",
        api_key=api_key,
        model=record.get("model") or "",
        trace_mode=trace_mode,
        runs_base_url=record.get("runs_base_url") or "",
    )


def get_agent_config(slot: str, fallback_name: str) -> AgentConfig:
    """Resolve the active agent for slot 'a' or 'b'. DB-first, env fallback."""
    slot_norm = (slot or "").strip().lower()
    if slot_norm not in {"a", "b"}:
        slot_norm = "a"
    with history_connect() as conn:
        record = config_store.load_active_agent_record(conn, slot_norm)  # type: ignore[arg-type]
    if record is not None:
        return _record_to_agent_config(record)
    prefix = f"AGENT_{slot_norm.upper()}"
    return _env_agent_config(prefix, fallback_name)


def get_judge_config() -> JudgeConfig:
    with history_connect() as conn:
        record = config_store.get_judge_record(conn)
    if record is not None:
        api_key_env = record.get("api_key_env") or ""
        api_key = os.getenv(api_key_env, "").strip() if api_key_env else ""
        return JudgeConfig(
            base_url=record.get("base_url") or "",
            api_key=api_key,
            model=record.get("model") or "",
        )
    return JudgeConfig(
        base_url=os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1").strip(),
        api_key=(
            os.getenv("JUDGE_OPENAI_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        ),
        model=os.getenv("JUDGE_MODEL", "").strip(),
    )


def get_timeout_seconds() -> float:
    raw = os.getenv("REQUEST_TIMEOUT_SECONDS", "180").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 180.0


def get_judge_timeout_seconds() -> float:
    raw = os.getenv("JUDGE_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return get_timeout_seconds()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return get_timeout_seconds()


def public_agent_config(agent: AgentConfig) -> dict[str, Any]:
    return {
        "name": agent.name,
        "base_url": agent.base_url,
        "model": agent.model,
        "configured": agent.configured,
        "has_api_key": bool(agent.api_key),
        "completion_path": agent.completion_url,
        "responses_path": agent.responses_url,
        "trace_mode": agent.trace_mode,
        "runs_path": agent.runs_url,
    }


def public_judge_config(judge: JudgeConfig) -> dict[str, Any]:
    return {
        "base_url": judge.base_url,
        "model": judge.model,
        "configured": judge.configured,
        "has_api_key": bool(judge.api_key),
        "completion_path": judge.completion_url,
    }


def extract_content(payload: dict[str, Any]) -> tuple[str, str | None]:
    choices = payload.get("choices") or []
    if not choices:
        return "", None
    first = choices[0] or {}
    finish_reason = first.get("finish_reason")
    message = first.get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content, finish_reason
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts), finish_reason
    return str(content), finish_reason


def extract_response_content(payload: dict[str, Any]) -> tuple[str, str | None]:
    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                parts.append(str(content.get("text") or ""))
    return "".join(parts), payload.get("status")


def truncate_text(value: Any, limit: int = MAX_EVENT_TEXT) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def truncate_log_text(value: Any, limit: int = 12000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[frontend log truncated {len(text) - limit} chars]"


def frontend_log_level(level: str) -> int:
    normalized = (level or "").strip().lower()
    if normalized in {"debug"}:
        return logging.DEBUG
    if normalized in {"info", "log"}:
        return logging.INFO
    if normalized in {"warn", "warning"}:
        return logging.WARNING
    return logging.ERROR


def update_item_tool_map(event_type: str, data: dict[str, Any], item_tool: dict[str, str]) -> None:
    """Track item_id → tool_name from response.output_item.added events so
    later function_call_arguments deltas can be attributed to a tool."""
    if event_type != "response.output_item.added":
        return
    item = data.get("item")
    if not isinstance(item, dict):
        return
    if item.get("type") != "function_call":
        return
    iid = item.get("id")
    name = item.get("name")
    if isinstance(iid, str) and isinstance(name, str):
        item_tool[iid] = name


def is_suppressed_function_call_args(event_type: str, data: dict[str, Any], item_tool: dict[str, str]) -> bool:
    """True for function_call_arguments stream events whose parent tool is in
    SUPPRESSED_ARG_TOOLS (e.g. report_markdown — args are redundant with the
    final output_text)."""
    if not event_type.startswith("response.function_call_arguments"):
        return False
    iid = data.get("item_id")
    if not isinstance(iid, str):
        return False
    return item_tool.get(iid, "") in SUPPRESSED_ARG_TOOLS


def normalize_trace_event(raw: dict[str, Any]) -> AgentTraceEvent:
    update = raw.get("update") if isinstance(raw.get("update"), dict) else raw.get("data")
    if not isinstance(update, dict):
        update = {}

    event_name = str(raw.get("event") or update.get("sessionUpdate") or "unknown")
    content = update.get("content")
    content_text = ""
    if isinstance(content, dict):
        content_text = str(content.get("text") or content.get("content") or "")
    elif isinstance(content, str):
        content_text = content

    item = update.get("item") if isinstance(update.get("item"), dict) else None
    item_type = str(item.get("type")) if item and isinstance(item.get("type"), str) else None
    tool_name = (
        raw.get("tool")
        or update.get("tool")
        or update.get("toolName")
        or update.get("name")
        or (item.get("name") if item and isinstance(item.get("name"), str) else None)
    )
    preview = raw.get("preview")
    if preview is None:
        preview = (
            update.get("preview")
            or update.get("status")
            or update.get("stopReason")
            or update.get("title")
        )

    text = raw.get("text") or update.get("text")
    delta = raw.get("delta") or update.get("delta")
    if event_name in {"agent_message_chunk", "thought_delta"} and content_text:
        delta = delta or content_text
    elif event_name in {"thought_start", "thought_done"} and content_text:
        text = text or content_text
    elif content_text:
        text = text or content_text

    err_raw = raw.get("error") if raw.get("error") is not None else update.get("error")
    if isinstance(err_raw, dict):
        err_raw = str(err_raw.get("message") or json.dumps(err_raw, ensure_ascii=False))
    elif err_raw is not None and not isinstance(err_raw, (bool, str)):
        err_raw = str(err_raw)

    return AgentTraceEvent(
        event=event_name,
        timestamp=raw.get("timestamp") or raw.get("created_at"),
        tool=tool_name,
        preview=truncate_text(preview) if preview is not None else None,
        duration=raw.get("duration") or update.get("duration"),
        error=err_raw,
        text=truncate_text(text) if text is not None else None,
        delta=truncate_text(delta) if delta is not None else None,
        item_type=item_type,
    )


def summarize_trace(events: list[AgentTraceEvent], supported: bool) -> dict[str, Any]:
    tool_call_count = sum(
        1 for event in events
        if (event.event in {"tool.started", "tool_call_update"} and event.preview != "completed")
        or (event.event == "response.output_item.added" and event.item_type == "function_call")
    )
    failed_tool_count = sum(
        1 for event in events
        if (event.event in {"tool.completed", "tool_call_update"} and bool(event.error))
        or (event.event == "response.output_item.done" and event.item_type == "function_call" and bool(event.error))
    )
    total_tool_duration_s = sum(
        event.duration or 0 for event in events if event.event in {"tool.completed", "tool_call_update"}
    )
    return {
        "supported": supported,
        "event_count": len(events),
        "tool_call_count": tool_call_count,
        "failed_tool_count": failed_tool_count,
        "total_tool_duration_s": round(total_tool_duration_s, 3),
        "reasoning_count": sum(1 for event in events if event.event in {"reasoning.available", "thought_delta", "thought_done"}),
        "message_delta_count": sum(1 for event in events if event.event in {"message.delta", "agent_message_chunk"}),
        "completion_event": next(
            (event.event for event in reversed(events) if event.event in {"run.completed", "run.failed"}),
            None,
        ),
    }


def build_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def format_request_error(stage: str, url: str, exc: Exception) -> str:
    return f"{stage} {url}: {type(exc).__name__}: {exc}"


def normalize_openclaw_url(base_url: str, path: str) -> str:
    clean_base = (base_url or "").strip().rstrip("/")
    clean_path = path if path.startswith("/") else f"/{path}"
    return f"{clean_base}{clean_path}"


def openclaw_credentials(agent: AgentConfig) -> tuple[str, str, bool]:
    """Return (email, password, should_register).

    OpenClaw's public API is cookie/session based. For production use, set
    OPENCLAW_EMAIL/OPENCLAW_PASSWORD or put "email:password" in the configured
    API-key env var. If neither is present, create an ephemeral test user.
    """
    raw = (agent.api_key or "").strip()
    if ":" in raw:
        email, password = raw.split(":", 1)
        if "@" in email and password:
            return email.strip(), password.strip(), False

    env_email = os.getenv("OPENCLAW_EMAIL", "").strip()
    env_password = os.getenv("OPENCLAW_PASSWORD", "").strip()
    if env_email and env_password:
        return env_email, env_password, False

    suffix = uuid.uuid4().hex[:16]
    return f"agent-arena-{suffix}@example.com", f"AgentArena{suffix}Aa1", True


def redact_openclaw_auth_text(text: str, email: str, password: str) -> str:
    redacted = text or ""
    if email:
        redacted = redacted.replace(email, "[redacted-email]")
    if password:
        redacted = redacted.replace(password, "[redacted-password]")
    return redacted


def parse_openclaw_sse_event(event_name: str, payload: dict[str, Any]) -> tuple[AgentTraceEvent | None, str, str | None, bool]:
    """Map one OpenClaw SSE event to AgentArena trace/content primitives."""
    name = event_name or "message"
    if name == "started":
        return (
            AgentTraceEvent(
                event="run.started",
                preview=str(payload.get("assistantMessageId") or ""),
                text=str(payload.get("runStartedAt") or ""),
            ),
            "",
            None,
            False,
        )
    if name == "status":
        phase = str(payload.get("phase") or payload.get("status") or "")
        detail = str(payload.get("detail") or "")
        text = detail or phase
        return (
            AgentTraceEvent(event="openclaw.status", preview=phase or None, text=text or None),
            "",
            None,
            False,
        )
    if name == "token":
        delta = str(payload.get("text") or "")
        return (
            AgentTraceEvent(event="message.delta", delta=delta),
            delta,
            None,
            False,
        )
    if name == "done":
        state = str(payload.get("runState") or "DONE")
        return (
            AgentTraceEvent(event="run.completed", preview=state),
            "",
            None,
            True,
        )
    if name in {"error", "failed"}:
        message = str(payload.get("message") or payload.get("error") or "OpenClaw run failed")
        return (
            AgentTraceEvent(event="run.failed", error=message, text=message),
            "",
            message,
            True,
        )
    return (
        AgentTraceEvent(event=f"openclaw.{name}", text=truncate_text(json.dumps(payload, ensure_ascii=False))),
        "",
        None,
        False,
    )


async def call_agent_openclaw(
    client: httpx.AsyncClient,
    agent: AgentConfig,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> AgentResult:
    if isinstance(client, httpx.AsyncClient):
        async with httpx.AsyncClient(timeout=client.timeout) as isolated_client:
            return await _call_agent_openclaw_with_client(
                isolated_client, agent, messages, temperature, max_tokens
            )
    return await _call_agent_openclaw_with_client(client, agent, messages, temperature, max_tokens)


async def _call_agent_openclaw_with_client(
    client: Any,
    agent: AgentConfig,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> AgentResult:
    del temperature, max_tokens  # OpenClaw's /api/chat endpoint does not expose these knobs.
    started = time.perf_counter()
    events: list[AgentTraceEvent] = []
    output_parts: list[str] = []
    session_id = ""
    try:
        email, password, should_register = openclaw_credentials(agent)
        if should_register:
            register_url = normalize_openclaw_url(agent.base_url, "/api/auth/register")
            register_response = await client.post(
                register_url,
                json={"email": email, "password": password},
                headers={"Content-Type": "application/json"},
            )
            if register_response.status_code >= 400:
                body = redact_openclaw_auth_text(register_response.text[:1000], email, password)
                raise RuntimeError(f"register HTTP {register_response.status_code}: {body}")

        csrf_url = normalize_openclaw_url(agent.base_url, "/api/auth/csrf")
        csrf_response = await client.get(csrf_url)
        if csrf_response.status_code >= 400:
            body = redact_openclaw_auth_text(csrf_response.text[:1000], email, password)
            raise RuntimeError(f"csrf HTTP {csrf_response.status_code}: {body}")
        csrf_token = str(csrf_response.json().get("csrfToken") or "")
        if not csrf_token:
            raise RuntimeError("csrf response did not include csrfToken")

        login_url = normalize_openclaw_url(agent.base_url, "/api/auth/callback/credentials")
        login_response = await client.post(
            login_url,
            data={
                "csrfToken": csrf_token,
                "email": email,
                "password": password,
                "redirect": "false",
                "json": "true",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if login_response.status_code >= 400:
            body = redact_openclaw_auth_text(login_response.text[:1000], email, password)
            raise RuntimeError(f"login HTTP {login_response.status_code}: {body}")

        session_url = normalize_openclaw_url(agent.base_url, "/api/sessions")
        session_response = await client.post(
            session_url,
            json={"title": "AgentArena compare"},
            headers={"Content-Type": "application/json"},
        )
        if session_response.status_code >= 400:
            body = redact_openclaw_auth_text(session_response.text[:1000], email, password)
            raise RuntimeError(f"session HTTP {session_response.status_code}: {body}")
        session_id = str(session_response.json().get("id") or "")
        if not session_id:
            raise RuntimeError("session response did not include id")

        system = "\n".join(msg["content"] for msg in messages if msg.get("role") == "system").strip()
        user_messages = [msg for msg in messages if msg.get("role") != "system"]
        user_input = user_messages[-1]["content"] if user_messages else ""
        message = f"{system}\n\n{user_input}".strip() if system else user_input

        event_name = ""
        data_lines: list[str] = []
        chat_url = normalize_openclaw_url(agent.base_url, "/api/chat")
        async with client.stream(
            "POST",
            chat_url,
            json={"sessionId": session_id, "message": message},
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise RuntimeError(f"chat HTTP {response.status_code}: {body.decode(errors='replace')[:1000]}")
            async for line in response.aiter_lines():
                if not line:
                    if not data_lines:
                        event_name = ""
                        continue
                    raw_data = "\n".join(data_lines)
                    data_lines = []
                    try:
                        payload = json.loads(raw_data)
                    except json.JSONDecodeError:
                        event_name = ""
                        continue
                    trace_event, delta, error, done = parse_openclaw_sse_event(event_name, payload)
                    if trace_event is not None and len(events) < MAX_TRACE_EVENTS:
                        events.append(trace_event)
                    if delta:
                        output_parts.append(delta)
                    if error:
                        latency_ms = int((time.perf_counter() - started) * 1000)
                        return AgentResult(
                            ok=False,
                            name=agent.name,
                            model=agent.model,
                            content="".join(output_parts),
                            latency_ms=latency_ms,
                            error=error,
                            raw_finish_reason="error",
                            trace_supported=True,
                            trace_events=events,
                            trace_summary=summarize_trace(events, True),
                        )
                    if done:
                        break
                    event_name = ""
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())

        final_output = "".join(output_parts)
        if not final_output.strip() and session_id:
            messages_url = normalize_openclaw_url(agent.base_url, f"/api/sessions/{session_id}/messages?limit=10")
            history_response = await client.get(messages_url)
            if history_response.status_code < 400:
                history = history_response.json()
                if isinstance(history, list):
                    for item in reversed(history):
                        if isinstance(item, dict) and str(item.get("role") or "").upper() == "ASSISTANT":
                            final_output = str(item.get("content") or "")
                            break

        latency_ms = int((time.perf_counter() - started) * 1000)
        effective_error = None if final_output.strip() else "empty output"
        return AgentResult(
            ok=effective_error is None,
            name=agent.name,
            model=agent.model,
            content=final_output,
            latency_ms=latency_ms,
            error=effective_error,
            raw_finish_reason="stop" if effective_error is None else "error",
            trace_supported=True,
            trace_events=events,
            trace_summary=summarize_trace(events, True),
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AgentResult(
            ok=False,
            name=agent.name,
            model=agent.model,
            content="".join(output_parts),
            latency_ms=latency_ms,
            error=format_request_error("openclaw failed", agent.base_url, exc),
            raw_finish_reason="error",
            trace_supported=True,
            trace_events=events,
            trace_summary=summarize_trace(events, True),
        )


async def call_agent_chat(
    client: httpx.AsyncClient,
    agent: AgentConfig,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> AgentResult:
    payload: dict[str, Any] = {
        "model": agent.model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    started = time.perf_counter()
    try:
        response = await client.post(
            agent.completion_url,
            json=payload,
            headers=build_headers(agent.api_key),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            body = response.text[:1000]
            return AgentResult(
                ok=False,
                name=agent.name,
                model=agent.model,
                latency_ms=latency_ms,
                error=f"HTTP {response.status_code}: {body}",
                trace_supported=False,
                trace_summary=summarize_trace([], False),
            )
        data = response.json()
        content, finish_reason = extract_content(data)
        return AgentResult(
            ok=True,
            name=agent.name,
            model=str(data.get("model") or agent.model),
            content=content,
            latency_ms=latency_ms,
            raw_finish_reason=finish_reason,
            trace_supported=False,
            trace_summary=summarize_trace([], False),
        )
    except Exception as exc:  # noqa: BLE001 - surface per-agent failures to the UI.
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AgentResult(
            ok=False,
            name=agent.name,
            model=agent.model,
            latency_ms=latency_ms,
            error=f"{type(exc).__name__}: {exc}",
            trace_supported=False,
            trace_summary=summarize_trace([], False),
        )


async def call_agent_responses(
    client: httpx.AsyncClient,
    agent: AgentConfig,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> AgentResult:
    started = time.perf_counter()
    system = "\n".join(msg["content"] for msg in messages if msg.get("role") == "system")
    user_messages = [msg for msg in messages if msg.get("role") != "system"]
    input_payload: Any
    if len(user_messages) > 1:
        input_payload = user_messages
    elif user_messages:
        input_payload = user_messages[-1]["content"]
    else:
        input_payload = ""

    payload: dict[str, Any] = {
        "model": agent.model,
        "input": input_payload,
        "stream": True,
    }
    if system:
        payload["instructions"] = system
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens

    parts: list[str] = []
    events: list[AgentTraceEvent] = []
    terminal: dict[str, Any] | None = None
    final_error: str | None = None
    item_tool: dict[str, str] = {}
    try:
        async with client.stream(
            "POST",
            agent.responses_url,
            json=payload,
            headers={**build_headers(agent.api_key), "Accept": "text/event-stream"},
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                latency_ms = int((time.perf_counter() - started) * 1000)
                return AgentResult(
                    ok=False,
                    name=agent.name,
                    model=agent.model,
                    latency_ms=latency_ms,
                    error=f"responses HTTP {response.status_code}: {body.decode(errors='replace')[:1000]}",
                    trace_supported=True,
                    trace_summary=summarize_trace([], True),
                )
            event_name = ""
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if not line:
                    if not data_lines:
                        event_name = ""
                        continue
                    raw_data = "\n".join(data_lines)
                    data_lines = []
                    if raw_data == "[DONE]":
                        break
                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        event_name = ""
                        continue
                    event_type = str(data.get("type") or event_name or "unknown")
                    if event_type == "response.output_text.delta" and data.get("delta"):
                        parts.append(str(data["delta"]))
                    elif event_type in {"response.completed", "response.failed", "response.incomplete"}:
                        terminal = data.get("response") if isinstance(data.get("response"), dict) else data
                        if event_type == "response.failed":
                            err = terminal.get("error") if isinstance(terminal, dict) else {}
                            final_error = str((err or {}).get("message") or "response failed")
                    update_item_tool_map(event_type, data, item_tool)
                    if is_suppressed_function_call_args(event_type, data, item_tool):
                        event_name = ""
                        continue
                    trace_event = normalize_trace_event({"event": event_type, "data": data})
                    if trace_event.event not in SUPPRESSED_TRACE_EVENTS and len(events) < MAX_TRACE_EVENTS:
                        events.append(trace_event)
                    event_name = ""
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AgentResult(
            ok=False,
            name=agent.name,
            model=agent.model,
            latency_ms=latency_ms,
            error=format_request_error("responses failed", agent.responses_url, exc),
            trace_supported=True,
            trace_events=events,
            trace_summary=summarize_trace(events, True),
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    content = "".join(parts)
    status = None
    if terminal:
        fallback, status = extract_response_content(terminal)
        content = content or fallback
    effective_error = final_error if final_error is not None else (
        "empty output" if not content.strip() else None
    )
    return AgentResult(
        ok=effective_error is None,
        name=agent.name,
        model=str((terminal or {}).get("model") or agent.model),
        content=content,
        latency_ms=latency_ms,
        error=effective_error,
        raw_finish_reason=status,
        trace_supported=True,
        trace_events=events,
        trace_summary=summarize_trace(events, True),
    )


async def collect_sse_events(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
) -> tuple[list[AgentTraceEvent], str, str | None]:
    events: list[AgentTraceEvent] = []
    final_output = ""
    final_error: str | None = None
    data_lines: list[str] = []

    async with client.stream("GET", url, headers=headers) as response:
        if response.status_code >= 400:
            body = await response.aread()
            raise RuntimeError(f"events HTTP {response.status_code}: {body.decode(errors='replace')[:1000]}")
        async for line in response.aiter_lines():
            if not line:
                if not data_lines:
                    continue
                raw_data = "\n".join(data_lines)
                data_lines = []
                try:
                    payload = json.loads(raw_data)
                except json.JSONDecodeError:
                    continue
                event = normalize_trace_event(payload)
                if event.event in SUPPRESSED_TRACE_EVENTS:
                    continue
                if len(events) < MAX_TRACE_EVENTS:
                    events.append(event)
                if event.event == "run.completed":
                    nested = payload.get("data") if isinstance(payload.get("data"), dict) else {}
                    final_output = str(payload.get("output") or nested.get("output") or "")
                    break
                if event.event == "run.failed":
                    nested = payload.get("data") if isinstance(payload.get("data"), dict) else {}
                    final_error = str(payload.get("error") or nested.get("error") or "run failed")
                    break
                continue
            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())

    return events, final_output, final_error


async def call_agent_runs(
    client: httpx.AsyncClient,
    agent: AgentConfig,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> AgentResult:
    started = time.perf_counter()
    system = "\n".join(msg["content"] for msg in messages if msg.get("role") == "system")
    user_messages = [msg for msg in messages if msg.get("role") != "system"]
    user_input = user_messages[-1]["content"] if user_messages else ""
    history = user_messages[:-1]
    payload: dict[str, Any] = {
        "model": agent.model,
        "input": user_input,
    }
    if system:
        payload["instructions"] = system
    if history:
        payload["conversation_history"] = history
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    runs_url = agent.runs_url
    try:
        response = await client.post(runs_url, json=payload, headers=build_headers(agent.api_key))
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AgentResult(
            ok=False,
            name=agent.name,
            model=agent.model,
            latency_ms=latency_ms,
            error=format_request_error("runs create failed", runs_url, exc),
            trace_supported=True,
            trace_summary=summarize_trace([], True),
        )

    try:
        if response.status_code >= 400:
            body = response.text[:1000]
            latency_ms = int((time.perf_counter() - started) * 1000)
            return AgentResult(
                ok=False,
                name=agent.name,
                model=agent.model,
                latency_ms=latency_ms,
                error=f"runs HTTP {response.status_code}: {body}",
                trace_supported=True,
                trace_summary=summarize_trace([], True),
            )
        run_id = response.json().get("run_id")
        if not run_id:
            raise RuntimeError("runs response did not include run_id")
        events_url = f"{runs_url.rstrip('/')}/{run_id}/events"
        try:
            events, final_output, final_error = await collect_sse_events(
                client,
                events_url,
                build_headers(agent.api_key),
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - started) * 1000)
            return AgentResult(
                ok=False,
                name=agent.name,
                model=agent.model,
                latency_ms=latency_ms,
                error=format_request_error("runs events failed", events_url, exc),
                trace_supported=True,
                trace_summary=summarize_trace([], True),
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        effective_error = final_error if final_error is not None else (
            "empty output" if not final_output.strip() else None
        )
        return AgentResult(
            ok=effective_error is None,
            name=agent.name,
            model=agent.model,
            content=final_output,
            latency_ms=latency_ms,
            error=effective_error,
            raw_finish_reason="stop" if effective_error is None else "error",
            trace_supported=True,
            trace_events=events,
            trace_summary=summarize_trace(events, True),
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AgentResult(
            ok=False,
            name=agent.name,
            model=agent.model,
            latency_ms=latency_ms,
            error=f"{type(exc).__name__}: {exc}",
            trace_supported=True,
            trace_summary=summarize_trace([], True),
        )


async def call_agent(
    client: httpx.AsyncClient,
    agent: AgentConfig,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> AgentResult:
    if not agent.configured:
        return AgentResult(
            ok=False,
            name=agent.name,
            model=agent.model,
            error="Agent is not configured. Set base URL and model in .env.",
            trace_supported=agent.trace_mode in {"runs", "openclaw"},
            trace_summary=summarize_trace([], agent.trace_mode in {"runs", "openclaw"}),
        )
    if agent.trace_mode == "openclaw":
        return await call_agent_openclaw(client, agent, messages, temperature, max_tokens)
    if agent.trace_mode == "runs":
        return await call_agent_runs(client, agent, messages, temperature, max_tokens)
    if agent.trace_mode == "responses":
        return await call_agent_responses(client, agent, messages, temperature, max_tokens)
    return await call_agent_chat(client, agent, messages, temperature, max_tokens)


async def call_agent_raw(
    client: httpx.AsyncClient,
    agent: AgentConfig,
    raw_body: dict[str, Any],
) -> AgentResult:
    """POST a user-supplied JSON body verbatim to the agent's /v1/responses URL.

    Forces stream=true so we can collect trace events; everything else is
    forwarded as-is.
    """
    if not agent.configured:
        return AgentResult(
            ok=False,
            name=agent.name,
            model=agent.model,
            error="Agent is not configured. Set base URL and model in .env.",
            trace_supported=True,
            trace_summary=summarize_trace([], True),
        )

    payload = dict(raw_body)
    payload["stream"] = True
    payload.setdefault("model", agent.model)

    started = time.perf_counter()
    parts: list[str] = []
    events: list[AgentTraceEvent] = []
    terminal: dict[str, Any] | None = None
    final_error: str | None = None
    item_tool: dict[str, str] = {}
    try:
        async with client.stream(
            "POST",
            agent.responses_url,
            json=payload,
            headers={**build_headers(agent.api_key), "Accept": "text/event-stream"},
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                latency_ms = int((time.perf_counter() - started) * 1000)
                return AgentResult(
                    ok=False,
                    name=agent.name,
                    model=str(payload.get("model") or agent.model),
                    latency_ms=latency_ms,
                    error=f"raw HTTP {response.status_code}: {body.decode(errors='replace')[:1000]}",
                    trace_supported=True,
                    trace_summary=summarize_trace([], True),
                )
            event_name = ""
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if not line:
                    if not data_lines:
                        event_name = ""
                        continue
                    raw_data = "\n".join(data_lines)
                    data_lines = []
                    if raw_data == "[DONE]":
                        break
                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        event_name = ""
                        continue
                    event_type = str(data.get("type") or event_name or "unknown")
                    if event_type == "response.output_text.delta" and data.get("delta"):
                        parts.append(str(data["delta"]))
                    elif event_type in {"response.completed", "response.failed", "response.incomplete"}:
                        terminal = data.get("response") if isinstance(data.get("response"), dict) else data
                        if event_type == "response.failed":
                            err = terminal.get("error") if isinstance(terminal, dict) else {}
                            final_error = str((err or {}).get("message") or "response failed")
                    update_item_tool_map(event_type, data, item_tool)
                    if is_suppressed_function_call_args(event_type, data, item_tool):
                        event_name = ""
                        continue
                    trace_event = normalize_trace_event({"event": event_type, "data": data})
                    if trace_event.event not in SUPPRESSED_TRACE_EVENTS and len(events) < MAX_TRACE_EVENTS:
                        events.append(trace_event)
                    event_name = ""
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").strip())
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AgentResult(
            ok=False,
            name=agent.name,
            model=str(payload.get("model") or agent.model),
            latency_ms=latency_ms,
            error=format_request_error("raw failed", agent.responses_url, exc),
            trace_supported=True,
            trace_events=events,
            trace_summary=summarize_trace(events, True),
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    content = "".join(parts)
    status: str | None = None
    if terminal:
        fallback, status = extract_response_content(terminal)
        content = content or fallback
    effective_error = final_error if final_error is not None else (
        "empty output" if not content.strip() else None
    )
    return AgentResult(
        ok=effective_error is None,
        name=agent.name,
        model=str((terminal or {}).get("model") or payload.get("model") or agent.model),
        content=content,
        latency_ms=latency_ms,
        error=effective_error,
        raw_finish_reason=status,
        trace_supported=True,
        trace_events=events,
        trace_summary=summarize_trace(events, True),
    )


def evaluate_assertion(content: str, assertion: BatchAssertion | None) -> bool | None:
    if assertion is None:
        return None
    target = content if assertion.case_sensitive else content.lower()
    needle = assertion.value if assertion.case_sensitive else assertion.value.lower()
    if assertion.type == "substring":
        return needle in target
    flags = 0 if assertion.case_sensitive else re.IGNORECASE
    try:
        return re.search(assertion.value, content, flags) is not None
    except re.error:
        return False


def _percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return int(round(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight))


BATCH_ITEM_TRACE_LIMIT = MAX_TRACE_EVENTS


def build_run_item(
    index: int,
    agent_key: Literal["a", "b"],
    agent: AgentConfig,
    result: AgentResult,
    assertion: BatchAssertion | None,
) -> BatchRunItem:
    trimmed_events = (result.trace_events or [])[:BATCH_ITEM_TRACE_LIMIT]
    return BatchRunItem(
        index=index,
        agent_key=agent_key,
        agent_name=agent.name,
        agent_model=result.model or agent.model,
        ok=result.ok,
        latency_ms=result.latency_ms,
        content=result.content,
        content_length=len(result.content or ""),
        finish_reason=result.raw_finish_reason,
        error=result.error,
        assertion_passed=evaluate_assertion(result.content, assertion) if result.ok else (False if assertion else None),
        trace_summary=result.trace_summary or {},
        trace_events=trimmed_events,
    )


def compute_batch_summary(
    items: list[BatchRunItem],
    agent_key: Literal["a", "b"],
    agent: AgentConfig,
    assertion: BatchAssertion | None,
) -> BatchAgentSummary:
    bucket = [item for item in items if item.agent_key == agent_key]
    total = len(bucket)
    success_items = [item for item in bucket if item.ok]
    success = len(success_items)
    latencies = [item.latency_ms for item in bucket if item.latency_ms is not None]
    content_lens = [item.content_length for item in success_items]
    tool_calls = [int((item.trace_summary or {}).get("tool_call_count") or 0) for item in success_items]
    failed_tools = sum(int((item.trace_summary or {}).get("failed_tool_count") or 0) for item in bucket)

    finish_reasons: dict[str, int] = {}
    for item in bucket:
        key = item.finish_reason or "<none>"
        finish_reasons[key] = finish_reasons.get(key, 0) + 1

    assertion_total = 0
    assertion_passed = 0
    if assertion is not None:
        for item in bucket:
            if item.assertion_passed is None:
                continue
            assertion_total += 1
            if item.assertion_passed:
                assertion_passed += 1

    return BatchAgentSummary(
        agent_key=agent_key,
        agent_name=agent.name,
        agent_model=agent.model,
        total=total,
        success=success,
        success_rate=round(success / total, 4) if total else 0.0,
        assertion_total=assertion_total,
        assertion_passed=assertion_passed,
        assertion_rate=(round(assertion_passed / assertion_total, 4) if assertion_total else None),
        latency_min_ms=min(latencies) if latencies else None,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p90_ms=_percentile(latencies, 90),
        latency_max_ms=max(latencies) if latencies else None,
        content_len_min=min(content_lens) if content_lens else None,
        content_len_avg=round(statistics.fmean(content_lens), 1) if content_lens else None,
        content_len_max=max(content_lens) if content_lens else None,
        tool_call_avg=round(statistics.fmean(tool_calls), 2) if tool_calls else None,
        failed_tool_total=failed_tools,
        finish_reasons=finish_reasons,
    )


def save_batch_history(response: BatchRunResponse) -> None:
    with history_connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO batch_runs (
                id, created_at, target, mode, iterations, concurrency, cancelled,
                request_json, summaries_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response.batch_id,
                response.created_at,
                response.request.target,
                response.request.mode,
                response.request.iterations,
                response.request.concurrency,
                int(response.cancelled),
                json_dumps(response.request),
                json_dumps(response.summaries),
            ),
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO batch_run_items (batch_id, idx, agent_key, ok, latency_ms, item_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    response.batch_id,
                    item.index,
                    item.agent_key,
                    int(item.ok),
                    item.latency_ms,
                    json_dumps(item),
                )
                for item in response.items
            ],
        )


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _agent_keys_for_target(target: BatchTarget) -> list[Literal["a", "b"]]:
    if target == "both":
        return ["a", "b"]
    return [target]


def _messages_for_form(form: BatchFormPayload) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if form.system.strip():
        messages.append({"role": "system", "content": form.system.strip()})
    messages.append({"role": "user", "content": form.input.strip()})
    return messages


async def stream_batch(request: BatchRunRequest):
    batch_id = new_id("batch")
    created_at = utc_now()
    keys = _agent_keys_for_target(request.target)
    agents: dict[str, AgentConfig] = {
        "a": get_agent_config("a", "Agent A"),
        "b": get_agent_config("b", "Agent B"),
    }
    total_items = request.iterations * len(keys)
    queue: asyncio.Queue[BatchRunItem | None] = asyncio.Queue()
    semaphore = asyncio.Semaphore(request.concurrency)
    cancelled = asyncio.Event()

    timeout = httpx.Timeout(get_timeout_seconds())
    client = httpx.AsyncClient(timeout=timeout)

    async def run_one(index: int, agent_key: Literal["a", "b"]) -> None:
        async with semaphore:
            if cancelled.is_set():
                return
            agent = agents[agent_key]
            if request.mode == "form":
                assert request.form is not None
                messages = _messages_for_form(request.form)
                result = await call_agent(
                    client,
                    agent,
                    messages,
                    request.form.temperature,
                    request.form.max_tokens,
                )
            else:
                assert request.raw_body is not None
                result = await call_agent_raw(client, agent, request.raw_body)
            item = build_run_item(index, agent_key, agent, result, request.assertion)
            await queue.put(item)

    async def scheduler() -> None:
        try:
            tasks: list[asyncio.Task[None]] = []
            for index in range(request.iterations):
                for key in keys:
                    tasks.append(asyncio.create_task(run_one(index, key)))
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                raise
        finally:
            await queue.put(None)

    scheduler_task = asyncio.create_task(scheduler())

    items: list[BatchRunItem] = []
    try:
        yield _sse_event(
            {
                "type": "batch.started",
                "batch_id": batch_id,
                "created_at": created_at,
                "total": total_items,
                "target": request.target,
                "mode": request.mode,
                "iterations": request.iterations,
                "concurrency": request.concurrency,
                "agents": {key: agents[key].name for key in keys},
            }
        )
        while True:
            item = await queue.get()
            if item is None:
                break
            items.append(item)
            yield _sse_event({"type": "run.completed", "item": model_to_dict(item)})

        summaries = {key: compute_batch_summary(items, key, agents[key], request.assertion) for key in keys}
        response_obj = BatchRunResponse(
            batch_id=batch_id,
            created_at=created_at,
            request=request,
            items=items,
            summaries=summaries,
            cancelled=False,
        )
        try:
            save_batch_history(response_obj)
        except Exception as exc:  # noqa: BLE001
            logging.exception("save_batch_history failed: %s", exc)
        yield _sse_event(
            {
                "type": "batch.completed",
                "batch_id": batch_id,
                "summaries": {k: model_to_dict(v) for k, v in summaries.items()},
                "items_count": len(items),
            }
        )
    except asyncio.CancelledError:
        cancelled.set()
        if not scheduler_task.done():
            scheduler_task.cancel()
        try:
            summaries = {key: compute_batch_summary(items, key, agents[key], request.assertion) for key in keys}
            partial = BatchRunResponse(
                batch_id=batch_id,
                created_at=created_at,
                request=request,
                items=items,
                summaries=summaries,
                cancelled=True,
            )
            save_batch_history(partial)
        except Exception as exc:  # noqa: BLE001
            logging.exception("save_batch_history (cancelled) failed: %s", exc)
        raise
    except Exception as exc:  # noqa: BLE001
        cancelled.set()
        if not scheduler_task.done():
            scheduler_task.cancel()
        yield _sse_event({"type": "batch.error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        cancelled.set()
        if not scheduler_task.done():
            scheduler_task.cancel()
            try:
                await scheduler_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await client.aclose()


def compact_trace_for_judge(result: AgentResult) -> dict[str, Any]:
    events = []
    for event in result.trace_events:
        if event.event == "message.delta":
            continue
        if hasattr(event, "model_dump"):
            events.append(event.model_dump(exclude_none=True))
        else:
            events.append(event.dict(exclude_none=True))
        if len(events) >= 40:
            break
    return {
        "trace_supported": result.trace_supported,
        "trace_summary": result.trace_summary,
        "events": events,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    clean = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", clean, re.DOTALL | re.IGNORECASE)
    if fence_match:
        clean = fence_match.group(1).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            return json.loads(clean[start:end + 1])
        raise


def build_judge_prompt(request: JudgeRequest) -> str:
    process_available = request.agent_a.trace_supported or request.agent_b.trace_supported
    return json.dumps(
        {
            "task": "请评估两个 Agent 对同一用户请求的回答。必须只返回 JSON，且 JSON 中所有自然语言内容必须使用简体中文。",
            "language": "简体中文",
            "important_output_rule": "winner 字段仍使用 a、b、tie；除字段名和枚举值外，summary、理由、优势、短板、建议都必须是中文。",
            "scoring_scale": "1-10, higher is better",
            "tie_rule": "如果最终加权总分差距小于 0.5，winner 必须为 tie",
            "final_answer_weight": "有过程事件时最终答案占 75%，否则最终答案占 100%",
            "process_weight": "只有存在过程事件时过程表现占 25%",
            "final_answer_dimensions": [
                "instruction_following",
                "correctness",
                "completeness",
                "actionability",
                "reasoning_quality",
                "clarity",
                "safety_and_risk_awareness",
            ],
            "process_dimensions": [
                "tool_relevance",
                "tool_efficiency",
                "evidence_use",
                "error_recovery",
                "process_transparency",
                "risk_control",
            ],
            "process_available": process_available,
            "required_json_shape": {
                "winner": "a|b|tie",
                "summary": "中文简短总评",
                "answer_scores": {"a": {}, "b": {}},
                "process_scores": {"available": process_available, "a": {}, "b": {}},
                "strengths": {"a": ["中文优势"], "b": ["中文优势"]},
                "weaknesses": {"a": ["中文短板"], "b": ["中文短板"]},
                "recommendations": {"a": ["中文建议"], "b": ["中文建议"]},
            },
            "user_input": request.input,
            "system_prompt": request.system,
            "agent_a": {
                "name": request.agent_a.name,
                "model": request.agent_a.model,
                "ok": request.agent_a.ok,
                "answer": request.agent_a.content,
                "error": request.agent_a.error,
                "process": compact_trace_for_judge(request.agent_a),
            },
            "agent_b": {
                "name": request.agent_b.name,
                "model": request.agent_b.model,
                "ok": request.agent_b.ok,
                "answer": request.agent_b.content,
                "error": request.agent_b.error,
                "process": compact_trace_for_judge(request.agent_b),
            },
        },
        ensure_ascii=False,
        indent=2,
    )


async def call_judge(request: JudgeRequest) -> JudgeResponse:
    judge = get_judge_config()
    if not judge.configured:
        return JudgeResponse(
            ok=False,
            error="Judge is not configured. Set JUDGE_MODEL and JUDGE_OPENAI_API_KEY or OPENAI_API_KEY in .env.",
        )
    if not (request.agent_a.content.strip() or request.agent_b.content.strip()):
        return JudgeResponse(ok=False, error="Both agent answers are empty; nothing to judge.")

    started = time.perf_counter()
    payload = {
        "model": judge.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个严格但公平的中文裁判。只评估提供的答案和公开过程事件，"
                    "不要要求或推断隐藏 chain-of-thought。必须只返回合法 JSON。"
                    "JSON 字段名可以保持英文，但所有解释性文本、理由、优势、短板和建议必须使用简体中文。"
                ),
            },
            {"role": "user", "content": build_judge_prompt(request)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(get_judge_timeout_seconds())) as client:
            response = await client.post(
                judge.completion_url,
                json=payload,
                headers=build_headers(judge.api_key),
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            return JudgeResponse(
                ok=False,
                latency_ms=latency_ms,
                error=f"HTTP {response.status_code}: {response.text[:1000]}",
            )
        data = response.json()
        content, _finish = extract_content(data)
        parsed = extract_json_object(content)
        return JudgeResponse(
            ok=True,
            winner=parsed.get("winner"),
            summary=str(parsed.get("summary") or ""),
            answer_scores=parsed.get("answer_scores") or {},
            process_scores=parsed.get("process_scores") or {},
            strengths=parsed.get("strengths") or {},
            weaknesses=parsed.get("weaknesses") or {},
            recommendations=parsed.get("recommendations") or {},
            latency_ms=latency_ms,
            raw=parsed,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return JudgeResponse(
            ok=False,
            latency_ms=latency_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


CONSISTENCY_MAX_SAMPLE_CHARS = 1200
CONSISTENCY_MAX_SAMPLES_PER_AGENT = 20


def build_consistency_prompt(
    user_input: str,
    items_by_agent: dict[str, list[dict[str, Any]]],
) -> str:
    return json.dumps(
        {
            "task": (
                "你是一个严谨的中文测试工程师。对同一个用户请求，下面提供了一个或两个 Agent "
                "在多次重复执行下的输出样本，请评估它们的稳定性，找出明显错误或前后不一致的样本，并给出修复建议。"
            ),
            "language": "简体中文",
            "scoring_rule": (
                "consistency_score 取 0-100：100 表示所有样本完全一致且无错误；"
                "70-90 表示主体一致但存在细节波动；50-70 表示存在明显分歧或部分错误；"
                "0-50 表示样本之间差异巨大或大量错误。"
            ),
            "stable_rule": "如果 consistency_score >= 80 且没有 high severity issue，stable 为 true，否则 false。",
            "issue_severity": "low / medium / high；high 表示这个样本显著错误或与其他样本严重不一致。",
            "required_json_shape": {
                "reports": [
                    {
                        "agent_key": "a|b",
                        "stable": True,
                        "consistency_score": 0,
                        "summary": "中文一句话总结这个 agent 的稳定性",
                        "issues": [
                            {"index": 0, "agent_key": "a", "problem": "中文描述问题", "severity": "low|medium|high"}
                        ],
                        "suggestions": ["中文修复建议"],
                    }
                ]
            },
            "user_input": user_input,
            "samples": items_by_agent,
        },
        ensure_ascii=False,
        indent=2,
    )


def collect_consistency_samples(
    items: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = item.get("agent_key") or ""
        if key not in {"a", "b"}:
            continue
        grouped.setdefault(key, []).append(
            {
                "index": item.get("index"),
                "ok": bool(item.get("ok")),
                "latency_ms": item.get("latency_ms"),
                "finish_reason": item.get("finish_reason"),
                "error": item.get("error"),
                "content": truncate_text(item.get("content") or "", CONSISTENCY_MAX_SAMPLE_CHARS),
            }
        )
    for key in list(grouped.keys()):
        bucket = grouped[key]
        bucket.sort(key=lambda x: x.get("index") or 0)
        if len(bucket) > CONSISTENCY_MAX_SAMPLES_PER_AGENT:
            grouped[key] = bucket[:CONSISTENCY_MAX_SAMPLES_PER_AGENT]
    return grouped


async def call_consistency_judge(
    batch_id: str,
    user_input: str,
    items: list[dict[str, Any]],
) -> ConsistencyResponse:
    judge = get_judge_config()
    created_at = utc_now()
    if not judge.configured:
        return ConsistencyResponse(
            ok=False,
            batch_id=batch_id,
            created_at=created_at,
            error="Judge is not configured. Set JUDGE_MODEL and JUDGE_OPENAI_API_KEY or OPENAI_API_KEY in .env.",
        )
    grouped = collect_consistency_samples(items)
    if not grouped:
        return ConsistencyResponse(
            ok=False,
            batch_id=batch_id,
            created_at=created_at,
            model=judge.model,
            error="No batch items available for consistency evaluation.",
        )

    started = time.perf_counter()
    payload = {
        "model": judge.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个严谨的中文测试工程师。专注于评估同一 Agent 在相同输入下多次输出的稳定性。"
                    "只能基于提供的样本判断，不要编造未提供的信息。必须只返回合法 JSON，"
                    "JSON 字段名可以保持英文，但所有解释、问题描述、建议都使用简体中文。"
                ),
            },
            {
                "role": "user",
                "content": build_consistency_prompt(user_input, grouped),
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(get_judge_timeout_seconds())) as client:
            response = await client.post(
                judge.completion_url,
                json=payload,
                headers=build_headers(judge.api_key),
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            return ConsistencyResponse(
                ok=False,
                batch_id=batch_id,
                created_at=created_at,
                model=judge.model,
                latency_ms=latency_ms,
                error=f"HTTP {response.status_code}: {response.text[:1000]}",
            )
        data = response.json()
        content, _finish = extract_content(data)
        parsed = extract_json_object(content)
        reports_raw = parsed.get("reports") or []
        agent_names = {
            key: (bucket[0].get("agent_name") if bucket else "")
            for key, bucket in (
                {k: [i for i in items if i.get("agent_key") == k] for k in ("a", "b")}
            ).items()
        }
        reports: list[ConsistencyAgentReport] = []
        for entry in reports_raw:
            if not isinstance(entry, dict):
                continue
            key = entry.get("agent_key")
            if key not in {"a", "b"}:
                continue
            issues_raw = entry.get("issues") or []
            issues: list[ConsistencyIssue] = []
            for issue in issues_raw:
                if not isinstance(issue, dict):
                    continue
                try:
                    issues.append(
                        ConsistencyIssue(
                            index=int(issue.get("index") or 0),
                            agent_key=key,
                            problem=str(issue.get("problem") or "").strip(),
                            severity=(issue.get("severity") if issue.get("severity") in {"low", "medium", "high"} else "medium"),
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue
            suggestions_raw = entry.get("suggestions") or []
            suggestions = [str(s).strip() for s in suggestions_raw if str(s).strip()]
            try:
                score = int(entry.get("consistency_score") or 0)
            except (TypeError, ValueError):
                score = 0
            score = max(0, min(100, score))
            reports.append(
                ConsistencyAgentReport(
                    agent_key=key,
                    agent_name=str(entry.get("agent_name") or agent_names.get(key, "") or f"Agent {key.upper()}"),
                    samples_evaluated=len(grouped.get(key, [])),
                    stable=bool(entry.get("stable")),
                    consistency_score=score,
                    summary=str(entry.get("summary") or "").strip(),
                    issues=issues,
                    suggestions=suggestions,
                )
            )
        return ConsistencyResponse(
            ok=True,
            batch_id=batch_id,
            created_at=created_at,
            model=judge.model,
            latency_ms=latency_ms,
            reports=reports,
            raw=parsed,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return ConsistencyResponse(
            ok=False,
            batch_id=batch_id,
            created_at=created_at,
            model=judge.model,
            latency_ms=latency_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


def save_consistency_result(response: ConsistencyResponse) -> str:
    result_id = new_id("consistency")
    with history_connect() as conn:
        conn.execute(
            """
            INSERT INTO batch_consistency_results (
                id, batch_id, created_at, model, ok, latency_ms, error, result_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                response.batch_id,
                response.created_at,
                response.model,
                int(response.ok),
                response.latency_ms,
                response.error,
                json_dumps(response),
            ),
        )
    return result_id


def list_consistency_results(batch_id: str) -> list[ConsistencyResponse]:
    with history_connect() as conn:
        rows = conn.execute(
            "SELECT result_json FROM batch_consistency_results WHERE batch_id = ? ORDER BY created_at DESC",
            (batch_id,),
        ).fetchall()
    out: list[ConsistencyResponse] = []
    for row in rows:
        try:
            out.append(ConsistencyResponse(**json.loads(row["result_json"])))
        except Exception:  # noqa: BLE001
            continue
    return out


app = FastAPI(title="AgentArena", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8787",
        "http://localhost:8787",
        "http://127.0.0.1:8683",
        "http://localhost:8683",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


def get_arena_api_key() -> str:
    return os.getenv("ARENA_API_KEY", "").strip()


def require_arena_auth(authorization: str | None = Header(default=None)) -> None:
    expected = get_arena_api_key()
    if not expected:
        return

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not secrets.compare_digest(token.strip(), expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


ArenaAuth = Depends(require_arena_auth)


@app.on_event("startup")
async def startup() -> None:
    init_history_db()
    with history_connect() as conn:
        config_store.migrate_from_env(conn)


@app.get("/api/config", dependencies=[ArenaAuth])
async def api_config() -> dict[str, Any]:
    return {
        "agents": {
            "a": public_agent_config(get_agent_config("a", "Agent A")),
            "b": public_agent_config(get_agent_config("b", "Agent B")),
        },
        "judge": public_judge_config(get_judge_config()),
        "timeout_seconds": get_timeout_seconds(),
        "judge_timeout_seconds": get_judge_timeout_seconds(),
        "auth_required": bool(get_arena_api_key()),
    }


@app.post("/api/compare", response_model=CompareResponse, dependencies=[ArenaAuth])
async def api_compare(request: CompareRequest) -> CompareResponse:
    user_input = request.input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="input cannot be empty")

    messages: list[dict[str, str]] = []
    if request.system.strip():
        messages.append({"role": "system", "content": request.system.strip()})
    messages.append({"role": "user", "content": user_input})

    agent_a = get_agent_config("a", "Agent A")
    agent_b = get_agent_config("b", "Agent B")
    timeout = httpx.Timeout(get_timeout_seconds())

    async with httpx.AsyncClient(timeout=timeout) as client:
        result_a, result_b = await asyncio.gather(
            call_agent(client, agent_a, messages, request.temperature, request.max_tokens),
            call_agent(client, agent_b, messages, request.temperature, request.max_tokens),
        )

    response = CompareResponse(
        run_id=new_id("run"),
        request={
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        },
        agents={"a": result_a, "b": result_b},
    )
    save_compare_history(response, user_input, request.system.strip(), request.temperature, request.max_tokens)
    return response


@app.post("/api/batch", dependencies=[ArenaAuth])
async def api_batch(request: BatchRunRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_batch(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/batch/{batch_id}", dependencies=[ArenaAuth])
async def api_batch_detail(batch_id: str) -> dict[str, Any]:
    with history_connect() as conn:
        row = conn.execute(
            "SELECT * FROM batch_runs WHERE id = ?",
            (batch_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="batch not found")
        item_rows = conn.execute(
            "SELECT item_json FROM batch_run_items WHERE batch_id = ? ORDER BY idx ASC, agent_key ASC",
            (batch_id,),
        ).fetchall()
    items = [json.loads(r["item_json"]) for r in item_rows]
    consistency = [model_to_dict(c) for c in list_consistency_results(batch_id)]
    return {
        "batch_id": row["id"],
        "created_at": row["created_at"],
        "target": row["target"],
        "mode": row["mode"],
        "iterations": row["iterations"],
        "concurrency": row["concurrency"],
        "cancelled": bool(row["cancelled"]),
        "request": json.loads(row["request_json"]),
        "summaries": json.loads(row["summaries_json"]),
        "items": items,
        "consistency_results": consistency,
    }


@app.post(
    "/api/batch/{batch_id}/consistency",
    response_model=ConsistencyResponse,
    dependencies=[ArenaAuth],
)
async def api_batch_consistency(batch_id: str) -> ConsistencyResponse:
    with history_connect() as conn:
        row = conn.execute(
            "SELECT * FROM batch_runs WHERE id = ?",
            (batch_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="batch not found")
        item_rows = conn.execute(
            "SELECT item_json FROM batch_run_items WHERE batch_id = ? ORDER BY idx ASC, agent_key ASC",
            (batch_id,),
        ).fetchall()
    items = [json.loads(r["item_json"]) for r in item_rows]
    request_data = json.loads(row["request_json"]) if row["request_json"] else {}
    form = request_data.get("form") or {}
    user_input = ""
    if isinstance(form, dict):
        user_input = str(form.get("input") or "").strip()
    if not user_input:
        raw_body = request_data.get("raw_body") or {}
        if isinstance(raw_body, dict):
            user_input = str(raw_body.get("input") or "").strip()
    response = await call_consistency_judge(batch_id, user_input, items)
    if response.ok:
        save_consistency_result(response)
    return response


@app.post("/api/judge", response_model=JudgeResponse, dependencies=[ArenaAuth])
async def api_judge(request: JudgeRequest) -> JudgeResponse:
    response = await call_judge(request)
    run_id = request.run_id or new_id("run")
    if not history_run_exists(run_id):
        save_compare_snapshot_from_judge_request(request, run_id)
    save_judge_history(run_id, response)
    return response


@app.get("/api/judge", dependencies=[ArenaAuth])
async def api_judge_help() -> dict[str, Any]:
    return {
        "ok": True,
        "message": "Use POST /api/judge with input, system, agent_a, and agent_b.",
        "judge": public_judge_config(get_judge_config()),
    }


@app.get("/api/history", response_model=HistoryListResponse, dependencies=[ArenaAuth])
async def api_history(limit: int = 50, offset: int = 0) -> HistoryListResponse:
    return list_history(limit, offset)


@app.get("/api/history/{run_id}", response_model=HistoryDetailResponse, dependencies=[ArenaAuth])
async def api_history_detail(run_id: str) -> HistoryDetailResponse:
    detail = get_history_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="history run not found")
    return detail


# ---------- Agent library CRUD + active pair --------------------------------


def _public_agent_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["api_key_env"] = config_store.env_var_status(record.get("api_key_env") or "")
    return out


@app.get("/api/agents", dependencies=[ArenaAuth])
async def api_list_agents() -> dict[str, Any]:
    with history_connect() as conn:
        records = config_store.list_agents(conn)
        pair = config_store.get_active_pair(conn)
    return {
        "agents": [_public_agent_record(r) for r in records],
        "active_pair": pair,
    }


@app.post("/api/agents", dependencies=[ArenaAuth])
async def api_create_agent(payload: AgentDefCreate) -> dict[str, Any]:
    with history_connect() as conn:
        rec = config_store.create_agent(
            conn,
            name=payload.name,
            base_url=payload.base_url,
            model=payload.model,
            trace_mode=payload.trace_mode,
            api_key_env=payload.api_key_env,
            runs_base_url=payload.runs_base_url,
            headers=payload.headers,
            description=payload.description,
        )
    return _public_agent_record(rec)


@app.patch("/api/agents/{agent_id}", dependencies=[ArenaAuth])
async def api_update_agent(agent_id: str, payload: AgentDefPatch) -> dict[str, Any]:
    with history_connect() as conn:
        rec = config_store.update_agent(
            conn,
            agent_id,
            name=payload.name,
            base_url=payload.base_url,
            model=payload.model,
            trace_mode=payload.trace_mode,
            api_key_env=payload.api_key_env,
            runs_base_url=payload.runs_base_url,
            headers=payload.headers,
            description=payload.description,
        )
    if rec is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return _public_agent_record(rec)


@app.delete("/api/agents/{agent_id}", dependencies=[ArenaAuth])
async def api_delete_agent(agent_id: str) -> dict[str, Any]:
    with history_connect() as conn:
        ok = config_store.delete_agent(conn, agent_id)
    if not ok:
        raise HTTPException(status_code=404, detail="agent not found")
    return {"ok": True}


@app.get("/api/agents/active-pair", dependencies=[ArenaAuth])
async def api_get_active_pair() -> dict[str, Any]:
    with history_connect() as conn:
        return config_store.get_active_pair(conn)


@app.put("/api/agents/active-pair", dependencies=[ArenaAuth])
async def api_set_active_pair(payload: ActivePairSet) -> dict[str, Any]:
    with history_connect() as conn:
        # Validate referenced agents exist (None is allowed = clear slot).
        for aid in (payload.a_agent_id, payload.b_agent_id):
            if aid and config_store.get_agent(conn, aid) is None:
                raise HTTPException(status_code=400, detail=f"agent {aid} not found")
        return config_store.set_active_pair(
            conn,
            a_agent_id=payload.a_agent_id,
            b_agent_id=payload.b_agent_id,
        )


# ---------- Judge config -----------------------------------------------------


@app.get("/api/judge/config", dependencies=[ArenaAuth])
async def api_get_judge_config() -> dict[str, Any]:
    with history_connect() as conn:
        rec = config_store.get_judge_record(conn)
    if rec is None:
        # Expose env-fallback shape so the frontend form has values to render.
        env_fallback = get_judge_config()
        return {
            "base_url": env_fallback.base_url,
            "model": env_fallback.model,
            "api_key_env": config_store.env_var_status(""),
            "source": "env",
        }
    out = dict(rec)
    out["api_key_env"] = config_store.env_var_status(rec.get("api_key_env") or "")
    out["source"] = "db"
    return out


@app.put("/api/judge/config", dependencies=[ArenaAuth])
async def api_set_judge_config(payload: JudgeConfigSet) -> dict[str, Any]:
    with history_connect() as conn:
        rec = config_store.set_judge_record(
            conn,
            base_url=payload.base_url,
            model=payload.model,
            api_key_env=payload.api_key_env,
        )
    out = dict(rec)
    out["api_key_env"] = config_store.env_var_status(rec.get("api_key_env") or "")
    out["source"] = "db"
    return out


# ---------- Datasets CRUD + cases CRUD --------------------------------------


@app.get("/api/datasets", dependencies=[ArenaAuth])
async def api_list_datasets() -> dict[str, Any]:
    with history_connect() as conn:
        return {"datasets": dataset_store.list_datasets(conn)}


@app.post("/api/datasets", dependencies=[ArenaAuth])
async def api_create_dataset(payload: DatasetCreate) -> dict[str, Any]:
    with history_connect() as conn:
        return dataset_store.create_dataset(
            conn, name=payload.name, description=payload.description
        )


@app.get("/api/datasets/{dataset_id}", dependencies=[ArenaAuth])
async def api_get_dataset(dataset_id: str) -> dict[str, Any]:
    with history_connect() as conn:
        ds = dataset_store.get_dataset(conn, dataset_id)
        if ds is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        ds["cases"] = dataset_store.list_cases(conn, dataset_id)
        ds["runs"] = dataset_store.list_runs(conn, dataset_id)
    return ds


@app.patch("/api/datasets/{dataset_id}", dependencies=[ArenaAuth])
async def api_update_dataset(dataset_id: str, payload: DatasetPatch) -> dict[str, Any]:
    with history_connect() as conn:
        ds = dataset_store.update_dataset(
            conn, dataset_id, name=payload.name, description=payload.description
        )
    if ds is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return ds


@app.delete("/api/datasets/{dataset_id}", dependencies=[ArenaAuth])
async def api_delete_dataset(dataset_id: str) -> dict[str, Any]:
    with history_connect() as conn:
        ok = dataset_store.delete_dataset(conn, dataset_id)
    if not ok:
        raise HTTPException(status_code=404, detail="dataset not found")
    return {"ok": True}


@app.post("/api/datasets/{dataset_id}/cases", dependencies=[ArenaAuth])
async def api_create_case(dataset_id: str, payload: DatasetCaseCreate) -> dict[str, Any]:
    with history_connect() as conn:
        if dataset_store.get_dataset(conn, dataset_id) is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        return dataset_store.create_case(
            conn,
            dataset_id=dataset_id,
            query=payload.query,
            system_prompt=payload.system_prompt,
            expected_answer=payload.expected_answer,
            tags=payload.tags,
            source_run_id=payload.source_run_id,
        )


@app.patch("/api/datasets/cases/{case_id}", dependencies=[ArenaAuth])
async def api_update_case(case_id: str, payload: DatasetCasePatch) -> dict[str, Any]:
    with history_connect() as conn:
        case = dataset_store.update_case(
            conn,
            case_id,
            query=payload.query,
            system_prompt=payload.system_prompt,
            expected_answer=payload.expected_answer,
            tags=payload.tags,
        )
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    return case


@app.delete("/api/datasets/cases/{case_id}", dependencies=[ArenaAuth])
async def api_delete_case(case_id: str) -> dict[str, Any]:
    with history_connect() as conn:
        ok = dataset_store.delete_case(conn, case_id)
    if not ok:
        raise HTTPException(status_code=404, detail="case not found")
    return {"ok": True}


# ---------- Harvest a run into a dataset + candidates listing ---------------


@app.get("/api/arena-runs/candidates", dependencies=[ArenaAuth])
async def api_arena_run_candidates(limit: int = 30) -> dict[str, Any]:
    """Recent arena_runs that have NOT yet been harvested into any dataset.

    Used by the dataset detail page's Candidates tab so users can promote
    interesting one-off arena runs into a permanent eval case.
    """
    safe_limit = max(1, min(limit, 100))
    with history_connect() as conn:
        rows = conn.execute(
            """
            SELECT r.id, r.created_at, r.input, r.system, r.agent_a_name, r.agent_b_name
            FROM arena_runs r
            WHERE NOT EXISTS (
                SELECT 1 FROM dataset_cases c WHERE c.source_run_id = r.id
            )
            ORDER BY r.created_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return {
        "candidates": [
            {
                "run_id": r["id"],
                "created_at": r["created_at"],
                "input": r["input"],
                "system": r["system"] or "",
                "agent_a_name": r["agent_a_name"] or "",
                "agent_b_name": r["agent_b_name"] or "",
            }
            for r in rows
        ]
    }


@app.post("/api/arena-runs/{run_id}/save-to-dataset", dependencies=[ArenaAuth])
async def api_save_run_to_dataset(run_id: str, payload: HarvestCasePayload) -> dict[str, Any]:
    with history_connect() as conn:
        run_row = conn.execute(
            "SELECT input, system FROM arena_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if run_row is None:
            raise HTTPException(status_code=404, detail="arena run not found")
        if dataset_store.get_dataset(conn, payload.dataset_id) is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        query = (payload.query_override or run_row["input"] or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is empty")
        system_prompt = (
            payload.system_override
            if payload.system_override is not None
            else (run_row["system"] or "")
        )
        case = dataset_store.create_case(
            conn,
            dataset_id=payload.dataset_id,
            query=query,
            system_prompt=system_prompt,
            expected_answer=payload.expected_answer,
            tags=payload.tags,
            source_run_id=run_id,
        )
    return case


@app.post("/api/batch-items/save-to-dataset", dependencies=[ArenaAuth])
async def api_save_batch_item_to_dataset(payload: BatchItemHarvestPayload) -> dict[str, Any]:
    # Batch items don't have an entry in arena_runs (different storage path),
    # so we resolve query/system from batch_runs.request_json and synthesize
    # a composite source_run_id of the form "batch:{batch_id}#{idx}/{agent_key}".
    with history_connect() as conn:
        batch_row = conn.execute(
            "SELECT request_json FROM batch_runs WHERE id = ?", (payload.batch_id,)
        ).fetchone()
        if batch_row is None:
            raise HTTPException(status_code=404, detail="batch not found")
        item_row = conn.execute(
            "SELECT 1 FROM batch_run_items WHERE batch_id = ? AND idx = ? AND agent_key = ?",
            (payload.batch_id, payload.idx, payload.agent_key),
        ).fetchone()
        if item_row is None:
            raise HTTPException(status_code=404, detail="batch item not found")
        if dataset_store.get_dataset(conn, payload.dataset_id) is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        try:
            request_data = json.loads(batch_row["request_json"]) or {}
        except (TypeError, ValueError):
            request_data = {}
        form = request_data.get("form") if isinstance(request_data.get("form"), dict) else {}
        raw_body = request_data.get("raw_body") if isinstance(request_data.get("raw_body"), dict) else {}
        default_query = form.get("input") or raw_body.get("input") or ""
        default_system = form.get("system") or raw_body.get("system") or ""
        query = (payload.query_override or default_query or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is empty")
        system_prompt = (
            payload.system_override
            if payload.system_override is not None
            else (default_system or "")
        )
        source_id = f"batch:{payload.batch_id}#{payload.idx}/{payload.agent_key}"
        case = dataset_store.create_case(
            conn,
            dataset_id=payload.dataset_id,
            query=query,
            system_prompt=system_prompt,
            expected_answer=payload.expected_answer,
            tags=payload.tags,
            source_run_id=source_id,
        )
    return case


# ---------- Dataset eval run: start (async), fetch, list --------------------


async def _orchestrate_dataset_run(
    *,
    run_id: str,
    dataset_id: str,
    case_ids: list[str] | None,
    judge_each: bool,
) -> None:
    """Background task: run every selected case through agents A and B, then
    judge each. Writes per-case rows into dataset_run_items, updates summary
    on dataset_runs at the end. Catches per-case exceptions so one bad case
    can't tank the whole run.
    """
    agent_a = get_agent_config("a", "Agent A")
    agent_b = get_agent_config("b", "Agent B")
    judge = get_judge_config()
    timeout = httpx.Timeout(get_timeout_seconds())

    with history_connect() as conn:
        all_cases = dataset_store.list_cases(conn, dataset_id)
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in all_cases if c["id"] in wanted]
    else:
        cases = all_cases

    summary: dict[str, Any] = {
        "total": len(cases),
        "completed": 0,
        "failed": 0,
        "judged": 0,
        "winners": {"A": 0, "B": 0, "tie": 0, "unknown": 0},
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        for idx, case in enumerate(cases):
            messages: list[dict[str, str]] = []
            if case["system_prompt"]:
                messages.append({"role": "system", "content": case["system_prompt"]})
            messages.append({"role": "user", "content": case["query"]})

            a_run_id_used: str | None = None
            b_run_id_used: str | None = None
            judge_result_id: str | None = None
            body: dict[str, Any] = {
                "case": {
                    "id": case["id"],
                    "query": case["query"],
                    "expected_answer": case["expected_answer"],
                }
            }
            item_status = "completed"
            try:
                result_a, result_b = await asyncio.gather(
                    call_agent(client, agent_a, messages, 0.2, None),
                    call_agent(client, agent_b, messages, 0.2, None),
                )
                compare = CompareResponse(
                    run_id=new_id("run"),
                    request={
                        "messages": messages,
                        "temperature": 0.2,
                        "max_tokens": None,
                        "dataset_run_id": run_id,
                        "dataset_case_id": case["id"],
                    },
                    agents={"a": result_a, "b": result_b},
                )
                save_compare_history(
                    compare, case["query"], case["system_prompt"], 0.2, None
                )
                a_run_id_used = compare.run_id
                b_run_id_used = compare.run_id
                body["agents"] = {
                    "a": agent_history_summary(result_a),
                    "b": agent_history_summary(result_b),
                }

                if judge_each and judge.configured:
                    judge_resp = await call_judge(
                        JudgeRequest(
                            run_id=compare.run_id,
                            input=case["query"],
                            system=case["system_prompt"],
                            agent_a=result_a,
                            agent_b=result_b,
                        )
                    )
                    if judge_resp.ok:
                        save_judge_history(compare.run_id, judge_resp)
                        judge_result_id = judge_resp.judge_id
                        summary["judged"] += 1
                        winner = (judge_resp.winner or "unknown").upper()
                        key = winner if winner in {"A", "B", "TIE"} else "UNKNOWN"
                        summary["winners"][key.lower() if key != "TIE" else "tie"] += 1
                    body["judge"] = {
                        "ok": judge_resp.ok,
                        "winner": judge_resp.winner,
                        "summary": judge_resp.summary,
                        "error": judge_resp.error,
                    }
                summary["completed"] += 1
            except Exception as exc:  # noqa: BLE001
                item_status = "failed"
                summary["failed"] += 1
                body["error"] = str(exc)

            with history_connect() as conn:
                dataset_store.upsert_run_item(
                    conn,
                    run_id=run_id,
                    case_id=case["id"],
                    idx=idx,
                    status=item_status,
                    a_run_id=a_run_id_used,
                    b_run_id=b_run_id_used,
                    judge_result_id=judge_result_id,
                    body=body,
                )

    with history_connect() as conn:
        dataset_store.finish_run(
            conn,
            run_id,
            status="completed" if summary["failed"] == 0 else "completed_with_errors",
            summary=summary,
        )


@app.post("/api/datasets/{dataset_id}/runs", dependencies=[ArenaAuth])
async def api_start_dataset_run(
    dataset_id: str, payload: DatasetRunStart
) -> dict[str, Any]:
    with history_connect() as conn:
        if dataset_store.get_dataset(conn, dataset_id) is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        pair = config_store.get_active_pair(conn)
        judge_rec = config_store.get_judge_record(conn)
        cases = dataset_store.list_cases(conn, dataset_id)
        if payload.case_ids:
            wanted = set(payload.case_ids)
            cases = [c for c in cases if c["id"] in wanted]
        if not cases:
            raise HTTPException(status_code=400, detail="no cases selected")
        run = dataset_store.create_run(
            conn,
            dataset_id=dataset_id,
            agent_a_id=pair.get("a_agent_id"),
            agent_b_id=pair.get("b_agent_id"),
            judge_model=(judge_rec or {}).get("model"),
        )

    asyncio.create_task(
        _orchestrate_dataset_run(
            run_id=run["id"],
            dataset_id=dataset_id,
            case_ids=payload.case_ids,
            judge_each=payload.judge_each,
        )
    )
    return run


@app.get("/api/datasets/{dataset_id}/runs", dependencies=[ArenaAuth])
async def api_list_dataset_runs(dataset_id: str) -> dict[str, Any]:
    with history_connect() as conn:
        if dataset_store.get_dataset(conn, dataset_id) is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        return {"runs": dataset_store.list_runs(conn, dataset_id)}


@app.get("/api/datasets/runs/{run_id}", dependencies=[ArenaAuth])
async def api_get_dataset_run(run_id: str) -> dict[str, Any]:
    with history_connect() as conn:
        run = dataset_store.get_run(conn, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        run["items"] = dataset_store.list_run_items(conn, run_id)
    return run


@app.post("/api/frontend-log", dependencies=[ArenaAuth])
async def api_frontend_log(request: FrontendLogRequest) -> dict[str, Any]:
    record = {
        "source": request.source,
        "timestamp": request.timestamp,
        "url": request.url,
        "user_agent": request.user_agent,
        "message": truncate_log_text(request.message, 4000),
        "stack": truncate_log_text(request.stack),
        "component_stack": truncate_log_text(request.component_stack),
        "payload": request.payload,
    }
    frontend_logger.log(
        frontend_log_level(request.level),
        json.dumps(record, ensure_ascii=False, default=str),
    )
    return {"ok": True, "log_path": str(ACTIVE_FRONTEND_LOG_PATH)}


@app.get("/api/frontend-logs", dependencies=[ArenaAuth])
async def api_frontend_logs(limit: int = 120) -> dict[str, Any]:
    safe_limit = max(1, min(limit, 500))
    if not ACTIVE_FRONTEND_LOG_PATH.exists():
        return {"path": str(ACTIVE_FRONTEND_LOG_PATH), "lines": []}
    lines = ACTIVE_FRONTEND_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"path": str(ACTIVE_FRONTEND_LOG_PATH), "lines": lines[-safe_limit:]}


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/")
async def index():
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse(
        """
        <html>
          <body style="font-family: system-ui; margin: 40px">
            <h1>AgentArena backend is running</h1>
            <p>Frontend is not built yet. Run <code>cd frontend && npm install && npm run dev</code>
            for development, or <code>npm run build</code> to serve the built app here.</p>
          </body>
        </html>
        """
    )


@app.get("/{path:path}")
async def spa_fallback(path: str):
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend is not built")
