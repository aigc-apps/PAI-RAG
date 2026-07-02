from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger
from pydantic import BaseModel
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.agent_config import (
    AgentConfigDocument,
    SetupConfig,
    apply_runtime_status,
    load_agent_config,
    mask_secrets,
    save_agent_config,
)
from app.config import get_settings
from app.deps import AppState, get_state
from agent.tools.defaults import build_default_registry
from agent.tools.builtin.install_skill import _install_skill_sync

router = APIRouter()


class YamlPayload(BaseModel):
    yaml: str


class SearchTestPayload(BaseModel):
    query: str = "OpenAI"
    num_results: int = 3


class SkillInstallPayload(BaseModel):
    source: Dict[str, Any]
    enable_for_agent: Optional[str] = None
    enable_after_build: bool = False
    overwrite: bool = False


def _path() -> str:
    return get_settings().config_path


def _runtime_doc(state: AppState, *, mask: bool = True) -> AgentConfigDocument:
    settings = get_settings()
    doc = apply_runtime_status(
        load_agent_config(settings.config_path),
        settings,
        state.router,
    )
    return mask_secrets(doc) if mask else doc


def _save_and_reload(path: str, doc: AgentConfigDocument, state: AppState) -> None:
    settings = get_settings()
    catalog = None
    if path == settings.models_path:
        try:
            from app.providers import ModelCatalog, ProviderRouter

            catalog = ModelCatalog(**doc.models)
            ProviderRouter(catalog)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid models config: {exc}") from exc
    save_agent_config(path, doc)
    state.registry = build_default_registry(settings, agent_config=doc)
    if catalog is not None and state.router is not None:
        state.router.reload(catalog)
    state.agent_config = apply_runtime_status(doc, settings, state.router)


def _preserve_masked_secrets(doc: AgentConfigDocument, current: AgentConfigDocument) -> None:
    current_by_id = {provider.id: provider for provider in current.providers}
    for provider in doc.providers:
        existing = current_by_id.get(provider.id)
        for key in ("api_key", "access_key_id", "access_key_secret", "security_token"):
            if provider.settings.get(key) != "********":
                continue
            if existing and existing.settings.get(key):
                provider.settings[key] = existing.settings[key]
            else:
                provider.settings.pop(key, None)


@router.get("/v1/setup")
async def get_setup(state: AppState = Depends(get_state)):
    doc = _runtime_doc(state)
    return JSONResponse(doc.model_dump(mode="json"))


@router.put("/v1/setup")
async def update_setup(payload: SetupConfig, state: AppState = Depends(get_state)):
    path = _path()
    doc = load_agent_config(path)
    doc.setup = payload
    if doc.setup.completed and not doc.setup.completed_at:
        doc.setup.completed_at = datetime.now(timezone.utc).isoformat()
    _save_and_reload(path, doc, state)
    return JSONResponse(_runtime_doc(state).model_dump(mode="json"))


@router.get("/v1/config")
async def get_agent_config(state: AppState = Depends(get_state)):
    return JSONResponse(_runtime_doc(state).model_dump(mode="json"))


@router.put("/v1/config")
async def update_agent_config(
    payload: AgentConfigDocument,
    state: AppState = Depends(get_state),
):
    path = _path()
    existing = load_agent_config(path)
    existing.setup = payload.setup
    existing.models = payload.models
    existing.skills = payload.skills
    existing.default_agent = payload.default_agent
    existing.agents = payload.agents
    existing.providers = payload.providers
    existing.capabilities = payload.capabilities
    _preserve_masked_secrets(existing, load_agent_config(path))
    _save_and_reload(path, existing, state)
    return JSONResponse(_runtime_doc(state).model_dump(mode="json"))


@router.get("/v1/config.yaml")
async def get_agent_config_yaml(state: AppState = Depends(get_state)):
    doc = _runtime_doc(state, mask=True)
    body = yaml.safe_dump(
        doc.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=True,
    )
    return Response(content=body, media_type="text/yaml")


@router.put("/v1/config.yaml")
async def update_agent_config_yaml(
    payload: YamlPayload,
    state: AppState = Depends(get_state),
):
    try:
        doc = AgentConfigDocument(**(yaml.safe_load(payload.yaml) or {}))
    except Exception as exc:
        logger.warning("PUT /v1/config.yaml rejected YAML: {}\n--- payload ---\n{}", exc, payload.yaml)
        raise HTTPException(status_code=400, detail=f"invalid config YAML: {exc}") from exc
    _preserve_masked_secrets(doc, load_agent_config(_path()))
    _save_and_reload(_path(), doc, state)
    return JSONResponse(_runtime_doc(state).model_dump(mode="json"))


@router.post("/v1/config/reload-env")
async def reload_env(state: AppState = Depends(get_state)):
    """Re-read .env into os.environ and rebuild runtime objects so env-derived
    settings take effect without a process restart.

    `load_dotenv` only runs once at import (app/lean_main.py), so editing .env
    while the server runs does nothing until this is called (or the process
    restarts). `override=True` makes changed .env values win over the stale
    os.environ entries populated at boot. After refreshing env, the tool
    registry (sandbox provider reads api_key/account_id from os.environ at
    construction), the LLM provider router, and the runtime agent config are
    rebuilt. Safe to call repeatedly."""
    from dotenv import load_dotenv
    load_dotenv(override=True)
    settings = get_settings()
    doc = load_agent_config(settings.config_path)
    # Refresh the LLM provider router too — its API keys are env-derived.
    try:
        from app.providers import ModelCatalog, ProviderRouter
        if doc.models:
            catalog = ModelCatalog(**doc.models)
            state.router = ProviderRouter(catalog, path=settings.models_path)
    except Exception as exc:
        logger.warning("reload-env: provider router rebuild skipped: {}", exc)
    state.registry = build_default_registry(settings, agent_config=doc)
    state.agent_config = apply_runtime_status(doc, settings, state.router)
    logger.info("reload-env: .env reloaded, registry + router + agent_config rebuilt")
    return JSONResponse({"ok": True})


@router.post("/v1/config/search/test")
async def test_search_provider(
    payload: SearchTestPayload,
    state: AppState = Depends(get_state),
):
    tool = state.registry.get("web_search")
    if tool is None:
        raise HTTPException(status_code=400, detail="web_search is not configured")
    output = await tool.fn(query=payload.query, num_results=payload.num_results)
    return JSONResponse({"ok": not output.startswith("web_search failed:"), "output": output})


@router.post("/v1/skills/uploads")
async def upload_skill_zip(
    file: UploadFile = File(...),
    x_admin: Optional[str] = Header(default=None),
):
    _require_admin_header(x_admin)
    doc = load_agent_config(_path())
    upload_root = Path(str(doc.skills.install.get("upload_root") or "./data/skill-uploads"))
    upload_root.mkdir(parents=True, exist_ok=True)
    upload_id = f"up_{uuid.uuid4().hex}"
    target = upload_root / f"{upload_id}.zip"
    size = 0
    max_bytes = int(doc.skills.install.get("max_download_mb") or 50) * 1024 * 1024
    try:
        with target.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail="skill archive is too large")
                fh.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return JSONResponse({
        "upload_id": upload_id,
        "filename": file.filename,
        "size": size,
    })


@router.post("/v1/skills/install")
async def install_skill(
    payload: SkillInstallPayload,
    state: AppState = Depends(get_state),
    x_admin: Optional[str] = Header(default=None),
):
    _require_admin_header(x_admin)
    settings = get_settings()
    path = _path()
    doc = load_agent_config(path)
    try:
        result = await asyncio.to_thread(
            _install_skill_sync,
            payload.source,
            Path(str(doc.skills.root)).expanduser(),
            dict(doc.skills.install or {}),
            settings.app_env.lower(),
            payload.overwrite,
            payload.enable_for_agent,
            payload.enable_after_build,
            Path(str(doc.skills.install.get("upload_root") or "./data/skill-uploads")).expanduser(),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    installed = [
        item for item in doc.skills.installed
        if item.get("id") != result.get("id")
    ]
    installed.append({
        "id": result.get("id"),
        "name": result.get("name"),
        "version": result.get("version"),
        "path": result.get("path"),
        "status": result.get("status"),
        "dependency_status": result.get("dependency_status"),
        "source": result.get("source"),
        "installed_at": datetime.now(timezone.utc).isoformat(),
    })
    doc.skills.installed = installed
    if (
        payload.enable_for_agent
        and result.get("status") == "ready"
        and result.get("id")
    ):
        for agent in doc.agents:
            if agent.id != payload.enable_for_agent:
                continue
            if result["id"] not in agent.skills.enabled:
                agent.skills.enabled.append(str(result["id"]))
            break
    _save_and_reload(path, doc, state)
    runtime = _runtime_doc(state)
    return JSONResponse({
        "ok": True,
        "result": result,
        "config": runtime.model_dump(mode="json"),
    })


def _require_admin_header(value: Optional[str]) -> None:
    if str(value or "").lower() not in {"1", "true", "yes", "admin"}:
        raise HTTPException(status_code=403, detail="admin permission required")
