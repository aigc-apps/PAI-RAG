from __future__ import annotations
import os
from typing import List, Optional
import yaml
from pydantic import BaseModel
from loguru import logger
from app.llm import LeanLLM


class ModelConfig(BaseModel):
    id: str
    provider: str = "openai"
    base_url: str
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None  # direct (fallback/tests); precedence over env
    context_window: int = 128000
    max_output_tokens: int = 8000
    supports_tools: bool = True
    supports_reasoning: bool = False
    temperature: Optional[float] = None

    def resolve_key(self) -> str:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    def has_usable_key(self) -> bool:
        # A model is usable if it has a direct key (even ""), no key requirement
        # (api_key_env unset -> keyless/local), or its env var resolves non-empty.
        if self.api_key is not None or not self.api_key_env:
            return True
        return bool(os.environ.get(self.api_key_env, ""))


class ModelCatalog(BaseModel):
    default_model: str
    models: List[ModelConfig]


def load_catalog(path: Optional[str], settings) -> ModelCatalog:
    """Parse `models.yaml` if present; otherwise synthesize a one-model catalog from
    the legacy Settings (openai_* + default_model) so env-only deployments still work."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return ModelCatalog(**data)
    return ModelCatalog(
        default_model=settings.default_model,
        models=[
            ModelConfig(
                id=settings.default_model,
                provider="openai",
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
            )
        ],
    )


class ProviderRouter:
    """Routes a requested model id to its provider config + a cached LeanLLM client.
    Catalog is reloadable; warm clients are preserved across a reload for unchanged
    models. Single-process: one router per AppState (config is GitOps/eventually
    consistent across instances; runtime-state sync is a separate concern)."""

    def __init__(self, catalog: ModelCatalog, path: Optional[str] = None):
        self._path = path
        self._configs: dict = {}
        self._clients: dict = {}
        self._default = ""
        self._apply(catalog)

    def _apply(self, catalog: ModelCatalog) -> None:
        old_configs = self._configs
        old_clients = self._clients
        new_configs: dict = {}
        for m in catalog.models:
            if not m.has_usable_key():
                logger.warning(f"model '{m.id}': required key env '{m.api_key_env}' unset; omitting")
                continue
            new_configs[m.id] = m
        self._configs = new_configs
        self._default = (
            catalog.default_model
            if catalog.default_model in new_configs
            else (next(iter(new_configs), ""))
        )
        # Preserve warm clients only for configs that are byte-for-byte unchanged.
        self._clients = {
            mid: client
            for mid, client in old_clients.items()
            if mid in new_configs and new_configs[mid] == old_configs.get(mid)
        }

    def get_config(self, model_id: str) -> ModelConfig:
        cfg = self._configs.get(model_id)
        if cfg is None:
            raise KeyError(model_id)
        return cfg

    def get_llm(self, model_id: str):
        if model_id in self._clients:
            return self._clients[model_id]
        cfg = self.get_config(model_id)
        key = cfg.resolve_key() or "EMPTY"  # AsyncOpenAI needs a non-empty string
        llm = LeanLLM(
            base_url=cfg.base_url,
            api_key=key,
            model=cfg.id,
            max_tokens=cfg.max_output_tokens,
            enable_thinking=cfg.supports_reasoning,
            temperature=cfg.temperature if cfg.temperature is not None else 0.7,
        )
        self._clients[model_id] = llm
        return llm

    def register_llm(self, model_id: str, llm) -> None:
        """Inject a client (test seam; also a warm-override)."""
        self._clients[model_id] = llm

    @property
    def default_model_id(self) -> str:
        return self._default

    def list_models(self) -> List[ModelConfig]:
        return list(self._configs.values())

    def reload(self, catalog: ModelCatalog) -> None:
        self._apply(catalog)

    def reload_from_disk(self, settings=None) -> None:
        from app.config import get_settings
        self._apply(load_catalog(self._path, settings or get_settings()))
