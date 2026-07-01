from __future__ import annotations
import os
from typing import List, Optional
import yaml
from pydantic import BaseModel
from app.llm import LeanLLM


class ModelSpec(BaseModel):
    """Per-model parameters nested under a provider. Connection-level params
    (base_url, api_key) live on the owning ProviderConfig, so a model only
    carries what actually varies within a provider's catalogue."""

    id: str
    context_window: int = 128000
    max_output_tokens: int = 8000
    supports_tools: bool = True
    supports_reasoning: bool = False
    temperature: Optional[float] = None


class ProviderConfig(BaseModel):
    """One OpenAI-compatible provider. Owns the connection params shared by
    every model under `models`. Mirrors the hermes `custom_providers[]` shape:
    name / base_url / api_key / models.

    Key presence is NOT validated at load — a provider stays in the catalog
    even when its `api_key_env` is unset, so users can keep several providers
    configured and switch between them. The key is resolved and checked only
    when a client for one of its models is actually built (`ProviderRouter.
    get_llm`)."""

    name: str
    base_url: str
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None  # direct (fallback/tests); precedence over env
    models: List[ModelSpec]

    def resolve_key(self) -> str:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""


class ModelConfig(BaseModel):
    """Flat, *resolved* view of a single model — produced by flattening a
    ProviderConfig + ModelSpec. This is the record the router indexes and
    builds LeanLLM clients from; it carries the provider's connection params
    copied in so equality comparison (warm-client preservation) is simple."""

    id: str
    provider: str
    base_url: str
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None
    context_window: int = 128000
    max_output_tokens: int = 8000
    supports_tools: bool = True
    supports_reasoning: bool = False
    temperature: Optional[float] = None

    @property
    def qualified_id(self) -> str:
        return f"{self.provider}/{self.id}"

    def resolve_key(self) -> str:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    @classmethod
    def from_provider(cls, p: ProviderConfig, m: ModelSpec) -> "ModelConfig":
        return cls(
            id=m.id,
            provider=p.name,
            base_url=p.base_url,
            api_key_env=p.api_key_env,
            api_key=p.api_key,
            context_window=m.context_window,
            max_output_tokens=m.max_output_tokens,
            supports_tools=m.supports_tools,
            supports_reasoning=m.supports_reasoning,
            temperature=m.temperature,
        )


class ModelCatalog(BaseModel):
    default_model: str
    providers: List[ProviderConfig]

    def flatten(self) -> List[ModelConfig]:
        out: List[ModelConfig] = []
        for p in self.providers:
            for m in p.models:
                out.append(ModelConfig.from_provider(p, m))
        return out


def load_catalog(path: Optional[str], settings) -> ModelCatalog:
    """Parse the unified config's `models:` section.

    The product config is a single YAML document. Model provider settings live
    under `models`, while agent/tool/provider capability settings live beside it.
    """
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data.get("models"), dict):
            raise ValueError(f"config '{path}' must contain a models section")
        return ModelCatalog(**data["models"])
    from app.agent_config import DEFAULT_DOCUMENT

    return ModelCatalog(**DEFAULT_DOCUMENT.models)


class ProviderRouter:
    """Routes a requested model id to its provider config + a cached LeanLLM
    client. Catalog is reloadable; warm clients are preserved across a reload
    for unchanged models. Single-process: one router per AppState (config is
    GitOps/eventually consistent across instances; runtime-state sync is a
    separate concern)."""

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
        seen_providers: set = set()
        for p in catalog.providers:
            if p.name in seen_providers:
                raise ValueError(
                    f"duplicate provider name '{p.name}' in model catalog; "
                    f"provider names must be unique"
                )
            seen_providers.add(p.name)
            # Keys are not checked here: a provider stays in the catalog even
            # with its api_key_env unset, so it remains selectable/switchable.
            # The key is validated only when one of its models is used
            # (get_llm) — i.e. only the *used* provider is validated.
            seen_model_ids: set = set()
            for m in p.models:
                if m.id in seen_model_ids:
                    raise ValueError(
                        f"duplicate model id '{m.id}' under provider '{p.name}'"
                    )
                seen_model_ids.add(m.id)
                cfg = ModelConfig.from_provider(p, m)
                new_configs[cfg.qualified_id] = cfg
        if not new_configs:
            raise ValueError("model catalog must define at least one model")
        if catalog.default_model not in new_configs:
            available = ", ".join(sorted(new_configs))
            raise ValueError(
                f"default_model '{catalog.default_model}' is not defined in "
                f"model catalog; available models: {available}"
            )
        self._configs = new_configs
        self._default = catalog.default_model
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
        key = cfg.resolve_key()
        if not key:
            if cfg.api_key_env:
                # A key was required for this provider but its env var is unset.
                # Validate the *used* provider at use time with a clear message
                # rather than dropping it from the catalog at load.
                raise RuntimeError(
                    f"provider '{cfg.provider}' requires env var "
                    f"'{cfg.api_key_env}' to be set (model '{cfg.id}')"
                )
            key = "EMPTY"  # keyless / local provider — AsyncOpenAI needs non-empty
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
