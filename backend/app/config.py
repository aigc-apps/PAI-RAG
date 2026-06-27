from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    default_model: str = "gpt-4o-mini"
    db_url: str = "sqlite+aiosqlite:///./data/agent.db"
    store_backend: str = "sql"   # "sql" | "memory"
    agent_name: str = "Aria"
    agent_role: str = "a general-purpose AI assistant"
    search_provider: str = "none"
    search_api_key: str = ""
    search_endpoint: str = ""
    skills_dir: str = ""
    models_path: str = "models.yaml"
    memory_inject_limit: int = 30


def get_settings() -> Settings:
    return Settings()
