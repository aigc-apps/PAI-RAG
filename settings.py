"""Application settings loaded from environment variables and .env."""
import ast
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)


def _str(name, default=""):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _bool(name, default=False):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    value = value.strip()
    if value.startswith("["):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list):
            raise ValueError(f"{name} must be a list")
        return [str(item) for item in parsed]
    return [item.strip() for item in value.split(",") if item.strip()]


# LLM backend
API_KEY = _str("API_KEY", _str("OPENAI_API_KEY", ""))
API_BASE = _str("API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = _str("MODEL", "qwen-plus")

# Capacity
MAX_TOKENS = _int("MAX_TOKENS", 8192)
MAX_TURNS = _int("MAX_TURNS", 40)
HISTORY_TRIM_TOKENS = _int("HISTORY_TRIM_TOKENS", 80000)
TIMEOUT = _int("TIMEOUT", 300)

# Workspaces and concurrency
WORKSPACE_ROOT = _str("WORKSPACE_ROOT", "./workspaces")
ENFORCE_WORKSPACE_FOR_SERVER = _bool("ENFORCE_WORKSPACE_FOR_SERVER", False)
MAX_GLOBAL_RUNS = _int("MAX_GLOBAL_RUNS", 0)
MAX_USER_RUNS = _int("MAX_USER_RUNS", 0)
SQLITE_JOURNAL_MODE = _str("SQLITE_JOURNAL_MODE", "DELETE")

# Runner and Redis
RUNNER_BACKEND = _str("RUNNER_BACKEND", "thread")
REDIS_URL = _str("REDIS_URL", "redis://127.0.0.1:6379/0")
CELERY_BROKER_URL = _str("CELERY_BROKER_URL", REDIS_URL)
RUN_EVENT_TTL_SECONDS = _int("RUN_EVENT_TTL_SECONDS", 24 * 60 * 60)
ASK_USER_TIMEOUT_SECONDS = _int("ASK_USER_TIMEOUT_SECONDS", 30 * 60)
RUN_IDLE_TIMEOUT_SECONDS = _int("RUN_IDLE_TIMEOUT_SECONDS", 60 * 60)
SSE_HEARTBEAT_SECONDS = _int("SSE_HEARTBEAT_SECONDS", 15)

# Memory
ENABLE_SHARED_MEMORY_FOR_USERS = _bool("ENABLE_SHARED_MEMORY_FOR_USERS", False)
ENABLE_LONG_TERM_MEMORY_FOR_USERS = _bool("ENABLE_LONG_TERM_MEMORY_FOR_USERS", False)

# HTTP backend
BACKEND_BASE_URL = _str("BACKEND_BASE_URL", "http://127.0.0.1:8000")
MODEL_ALIASES = _list("MODEL_ALIASES", ["mini-agent", "agent"])
MAX_REQUEST_BODY_BYTES = _int("MAX_REQUEST_BODY_BYTES", 8 * 1024 * 1024)
SERVER_CORS_ORIGINS = _list("SERVER_CORS_ORIGINS", ["*"])
EXPOSE_SESSION_DEBUG = _bool("EXPOSE_SESSION_DEBUG", False)
