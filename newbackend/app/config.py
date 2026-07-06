from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    default_model: str = "openai/gpt-4o-mini"
    db_url: str = "sqlite+aiosqlite:///./data/agent.db"
    store_backend: str = "sql"   # "sql" | "memory"
    agent_name: str = "Aria"
    agent_role: str = "a general-purpose AI assistant"
    search_provider: str = "none"
    search_api_key: str = ""
    search_endpoint: str = ""
    skills_dir: str = ""
    config_path: str = "./data/config.yaml"
    models_path: str = "./data/config.yaml"
    app_env: str = "development"
    memory_inject_limit: int = 30
    memory_enabled: bool = False
    memory_model: str = ""
    summary_enabled: bool = False
    summary_keep_recent: int = 20
    summary_batch: int = 20
    project_context: str = ""
    # Sandbox file artifacts. `files_url_secret` signs the /v1/files tokens
    # (feature is off / fails closed when empty). `files_nas_local_root` is the
    # backend host's mount of the same NAS export the sandbox sees at /mnt/user;
    # empty => serve by reading bytes back from the live sandbox instead.
    files_url_secret: str = ""
    files_nas_local_root: str = ""
    files_max_bytes: int = 25 * 1024 * 1024


def get_settings() -> Settings:
    return Settings()
