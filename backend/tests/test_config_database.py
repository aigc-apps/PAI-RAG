from __future__ import annotations

from sqlalchemy.engine import make_url

from app.config import Settings


DB_ENV_KEYS = ("DB_URL", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")


def _clear_db_env(monkeypatch) -> None:
    for key in DB_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_separate_database_env_builds_url_with_raw_password(monkeypatch):
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("DB_HOST", "pgm.example.com")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_NAME", "loop0713")
    monkeypatch.setenv("DB_USER", "pairag")
    monkeypatch.setenv("DB_PASSWORD", "Test1234@/%:#")

    settings = Settings(_env_file=None)
    parsed = make_url(settings.db_url)

    assert parsed.drivername == "postgresql+asyncpg"
    assert parsed.host == "pgm.example.com"
    assert parsed.port == 5432
    assert parsed.database == "loop0713"
    assert parsed.username == "pairag"
    assert parsed.password == "Test1234@/%:#"


def test_db_url_takes_precedence_over_separate_database_env(monkeypatch):
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("DB_URL", "sqlite+aiosqlite:///./explicit.db")
    monkeypatch.setenv("DB_HOST", "ignored.example.com")
    monkeypatch.setenv("DB_NAME", "ignored")
    monkeypatch.setenv("DB_USER", "ignored")
    monkeypatch.setenv("DB_PASSWORD", "ignored")

    settings = Settings(_env_file=None)

    assert settings.db_url == "sqlite+aiosqlite:///./explicit.db"


def test_offline_pipeline_settings_are_bounded(monkeypatch):
    _clear_db_env(monkeypatch)
    settings = Settings(
        _env_file=None,
        sync_fetch_concurrency=0,
        sync_fetch_queue_size=10_000,
        sync_embedding_concurrency=100,
        sync_sql_batch_documents=0,
        sync_sql_batch_chunks=100_000,
        sync_es_bulk_target_bytes=1,
        sync_es_bulk_max_bytes=1,
        sync_progress_interval_seconds=0,
        job_heartbeat_seconds=0,
        job_lease_seconds=1,
    )

    assert settings.sync_fetch_concurrency == 1
    assert settings.sync_fetch_queue_size == 1_000
    assert settings.sync_embedding_concurrency == 16
    assert settings.sync_sql_batch_documents == 1
    assert settings.sync_sql_batch_chunks == 10_000
    assert settings.sync_es_bulk_target_bytes == 1024 * 1024
    assert settings.sync_es_bulk_max_bytes == 1024 * 1024
    assert settings.sync_progress_interval_seconds == 1.0
    assert settings.job_heartbeat_seconds == 1.0
    assert settings.job_lease_seconds == 3.0
