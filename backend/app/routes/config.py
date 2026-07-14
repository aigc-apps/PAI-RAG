from __future__ import annotations

import asyncio
import inspect
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger
from pydantic import BaseModel
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.agent_config import (
    AgentConfigDocument,
    InstalledSkill,
    SetupConfig,
    apply_runtime_status,
    authored_config_dict,
    load_agent_config,
    mask_secrets,
    remove_installed_skill,
    save_agent_config,
    set_agent_skill_enabled,
)
from app.auth import require_admin
from app.config import get_settings
from app.deps import AppState, get_state, rebuild_app_state_from_config
from app.store.base import User
from agent.custom_skills import _normalize_skill_id, skill_local_root
from agent.tools.builtin.install_skill import _install_skill_sync

router = APIRouter()


class YamlPayload(BaseModel):
    yaml: str


class SearchTestPayload(BaseModel):
    query: str = "OpenAI"
    num_results: int = 3


class ModelTestPayload(BaseModel):
    # A `provider/model-id` catalog ref; omit to test the deployment default LLM.
    model: Optional[str] = None


class SkillInstallPayload(BaseModel):
    source: Dict[str, Any]
    enable_for_agent: Optional[str] = None
    enable_after_build: bool = False
    overwrite: bool = False


class SkillEnablePayload(BaseModel):
    skill_id: str
    agent_id: Optional[str] = None
    enabled: bool = True


class SkillUninstallPayload(BaseModel):
    skill_id: str
    # A destructive, irreversible operation (removes files + drops the record +
    # unassigns from every agent), so the client must opt in explicitly.
    confirm: bool = False


async def _authored_doc(state: AppState) -> AgentConfigDocument:
    if getattr(state, "config_store", None) is not None:
        stored = await state.config_store.load()
        state.config_revision = stored.revision
        return stored.doc
    return load_agent_config(get_settings().config_path)


def _runtime_from_doc(
    state: AppState,
    doc: AgentConfigDocument,
    *,
    mask: bool = True,
) -> AgentConfigDocument:
    settings = get_settings()
    runtime = apply_runtime_status(doc, settings, state.router)
    return mask_secrets(runtime) if mask else runtime


async def _runtime_doc(state: AppState, *, mask: bool = True) -> AgentConfigDocument:
    return _runtime_from_doc(state, await _authored_doc(state), mask=mask)


async def _save_and_reload(
    doc: AgentConfigDocument,
    state: AppState,
    *,
    updated_by: Optional[str] = None,
) -> None:
    settings = get_settings()
    try:
        from app.providers import ModelCatalog, ProviderRouter

        catalog = ModelCatalog(**doc.models)
        ProviderRouter(catalog)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid models config: {exc}") from exc
    if getattr(state, "config_store", None) is not None:
        stored = await state.config_store.save(doc, updated_by=updated_by)
        doc = stored.doc
        state.config_revision = stored.revision
    else:
        save_agent_config(settings.config_path, doc)
    if state.router is not None:
        state.router.reload(catalog)
    # Rebuild the registry + knowledge search engine + runtime agent_config through
    # the single canonical reloader. A partial build_default_registry() call here used
    # to omit knowledge_service (and skip the search-engine rebuild), so every config
    # save silently dropped the KB tools until a process restart. Router is reloaded
    # first so runtime status sees the fresh catalog.
    rebuild_app_state_from_config(state, settings, doc)


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
    # knowledgebase.vectordb is a typed section (not a provider); restore masked
    # ES secrets from the on-disk config so a re-save of "********" doesn't blank them.
    vdb, cur_vdb = doc.knowledgebase.vectordb, current.knowledgebase.vectordb
    for key in ("api_key", "password"):
        if getattr(vdb, key) == "********":
            setattr(vdb, key, getattr(cur_vdb, key, "") or "")


@router.get("/v1/setup")
async def get_setup(state: AppState = Depends(get_state),
                    admin: User = Depends(require_admin)):
    doc = await _runtime_doc(state)
    return JSONResponse(doc.model_dump(mode="json"))


@router.put("/v1/setup")
async def update_setup(payload: SetupConfig, state: AppState = Depends(get_state),
                       admin: User = Depends(require_admin)):
    doc = await _authored_doc(state)
    doc.setup = payload
    if doc.setup.completed and not doc.setup.completed_at:
        doc.setup.completed_at = datetime.now(timezone.utc).isoformat()
    await _save_and_reload(doc, state, updated_by=admin.id)
    return JSONResponse((await _runtime_doc(state)).model_dump(mode="json"))


@router.get("/v1/config")
async def get_agent_config(state: AppState = Depends(get_state),
                           admin: User = Depends(require_admin)):
    return JSONResponse((await _runtime_doc(state)).model_dump(mode="json"))


@router.put("/v1/config")
async def update_agent_config(
    payload: AgentConfigDocument,
    state: AppState = Depends(get_state),
    admin: User = Depends(require_admin),
):
    existing = await _authored_doc(state)
    existing.setup = payload.setup
    existing.models = payload.models
    existing.knowledgebase = payload.knowledgebase
    existing.skills = payload.skills
    existing.default_instructions = payload.default_instructions
    existing.default_agent = payload.default_agent
    existing.agents = payload.agents
    existing.providers = payload.providers
    existing.capabilities = payload.capabilities
    _preserve_masked_secrets(existing, await _authored_doc(state))
    await _save_and_reload(existing, state, updated_by=admin.id)
    return JSONResponse((await _runtime_doc(state)).model_dump(mode="json"))


@router.get("/v1/config.yaml")
async def get_agent_config_yaml(state: AppState = Depends(get_state),
                                admin: User = Depends(require_admin)):
    doc = await _runtime_doc(state, mask=True)
    body = yaml.safe_dump(
        authored_config_dict(doc),
        sort_keys=False,
        allow_unicode=True,
    )
    return Response(content=body, media_type="text/yaml")


@router.put("/v1/config.yaml")
async def update_agent_config_yaml(
    payload: YamlPayload,
    state: AppState = Depends(get_state),
    admin: User = Depends(require_admin),
):
    try:
        doc = AgentConfigDocument(**(yaml.safe_load(payload.yaml) or {}))
    except Exception as exc:
        logger.warning("PUT /v1/config.yaml rejected YAML: {}\n--- payload ---\n{}", exc, payload.yaml)
        raise HTTPException(status_code=400, detail=f"invalid config YAML: {exc}") from exc
    _preserve_masked_secrets(doc, await _authored_doc(state))
    await _save_and_reload(doc, state, updated_by=admin.id)
    return JSONResponse((await _runtime_doc(state)).model_dump(mode="json"))


@router.post("/v1/config/reload-env")
async def reload_env(state: AppState = Depends(get_state),
                     admin: User = Depends(require_admin)):
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
    doc = await _authored_doc(state)
    # Refresh the LLM provider router too — its API keys are env-derived.
    try:
        from app.providers import ModelCatalog, ProviderRouter
        if doc.models:
            catalog = ModelCatalog(**doc.models)
            state.router = ProviderRouter(catalog, path=settings.models_path)
    except Exception as exc:
        logger.warning("reload-env: provider router rebuild skipped: {}", exc)
    # Rebuild registry + agent_config through the canonical reloader so the sandbox
    # provider picks up refreshed env creds AND knowledge_service stays wired — the
    # partial rebuild here previously dropped the KB tools on every reload-env.
    rebuild_app_state_from_config(state, settings, doc)
    logger.info("reload-env: .env reloaded, registry + router + agent_config rebuilt")
    return JSONResponse({"ok": True})


@router.post("/v1/config/search/test")
async def test_search_provider(
    payload: SearchTestPayload,
    state: AppState = Depends(get_state),
    admin: User = Depends(require_admin),
):
    tool = state.registry.get("web_search")
    if tool is None:
        raise HTTPException(status_code=400, detail="web_search is not configured")
    output = await tool.fn(query=payload.query, num_results=payload.num_results)
    return JSONResponse({"ok": not output.startswith("web_search failed:"), "output": output})


@router.post("/v1/config/models/test")
async def test_model_connection(
    payload: ModelTestPayload,
    state: AppState = Depends(get_state),
    admin: User = Depends(require_admin),
):
    """Probe an LLM connection end-to-end: resolve its key env, open the client,
    and stream a one-token completion. Tests the *saved* catalog (save the
    connection first), mirroring how the search test hits the live registry.
    Never raises on a bad connection — returns ``{ok: false, output}`` so the UI
    can show the reason inline."""
    router_ = getattr(state, "router", None)
    if router_ is None:
        return JSONResponse({"ok": False, "output": "No model catalog is configured."})
    model_id = (payload.model or getattr(router_, "default_model_id", "") or "").strip()
    if not model_id:
        return JSONResponse(
            {"ok": False, "output": "No model given and no default model is set."}
        )
    try:
        llm = router_.get_llm(model_id)
    except Exception as exc:  # unknown ref, wrong type, or unset key env
        return JSONResponse({"ok": False, "output": str(exc)})
    try:
        stream = llm.astream([{"role": "user", "content": "ping"}], max_tokens=1)
        if inspect.isawaitable(stream):
            stream = await stream
        async for chunk in stream:
            err = getattr(chunk, "error_message", None)
            if err:
                return JSONResponse({"ok": False, "output": f"{model_id}: {err}"})
    except Exception as exc:
        return JSONResponse({"ok": False, "output": f"{model_id}: {exc}"})
    return JSONResponse({"ok": True, "output": f"{model_id} responded."})


@router.post("/v1/skills/uploads")
async def upload_skill_zip(
    file: UploadFile = File(...),
    state: AppState = Depends(get_state),
    admin: User = Depends(require_admin),
):
    doc = await _authored_doc(state)
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
    admin: User = Depends(require_admin),
):
    settings = get_settings()
    doc = await _authored_doc(state)
    try:
        result = await asyncio.to_thread(
            _install_skill_sync,
            payload.source,
            Path(str(getattr(settings, "skill_local_root", "") or skill_local_root())).expanduser(),
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
        if item.id != result.get("id")
    ]
    installed.append(InstalledSkill(
        id=str(result.get("id")),
        name=str(result.get("name") or ""),
        version=str(result.get("version") or "0.0.0"),
        path=str(result.get("path") or ""),
        status=str(result.get("status") or "ready"),
        dependency_status=str(result.get("dependency_status") or "none"),
        source=result.get("source") if isinstance(result.get("source"), dict) else {},
        installed_at=datetime.now(timezone.utc).isoformat(),
    ))
    doc.skills.installed = installed
    if (
        payload.enable_for_agent
        and result.get("status") == "ready"
        and result.get("id")
    ):
        runtime = apply_runtime_status(doc, settings, state.router)
        try:
            set_agent_skill_enabled(
                doc,
                agent_id=str(payload.enable_for_agent),
                skill_id=str(result["id"]),
                enabled=True,
                installed=runtime.skills.installed,
            )
        except ValueError as exc:
            logger.warning("install: enable_for_agent skipped: {}", exc)
    await _save_and_reload(doc, state, updated_by=admin.id)
    runtime = await _runtime_doc(state)
    return JSONResponse({
        "ok": True,
        "result": result,
        "config": runtime.model_dump(mode="json"),
    })


@router.post("/v1/skills/enable")
async def enable_skill_for_agent(
    payload: SkillEnablePayload,
    state: AppState = Depends(get_state),
    admin: User = Depends(require_admin),
):
    settings = get_settings()
    doc = await _authored_doc(state)
    target_agent = payload.agent_id or doc.default_agent
    runtime = apply_runtime_status(doc, settings, state.router)
    try:
        result = set_agent_skill_enabled(
            doc,
            agent_id=str(target_agent),
            skill_id=payload.skill_id,
            enabled=payload.enabled,
            installed=runtime.skills.installed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _save_and_reload(doc, state, updated_by=admin.id)
    return JSONResponse({
        "ok": True,
        "result": result,
        "config": (await _runtime_doc(state)).model_dump(mode="json"),
    })


@router.post("/v1/skills/uninstall")
async def uninstall_skill(
    payload: SkillUninstallPayload,
    state: AppState = Depends(get_state),
    admin: User = Depends(require_admin),
):
    """Admin-only, confirmation-gated removal of an installed skill: deletes its
    on-disk package dir (under the local root), drops its ``skills.installed``
    record, and unassigns it from every agent's ``skills.enabled``."""
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="uninstall requires confirm=true")
    settings = get_settings()
    doc = await _authored_doc(state)
    normalized = _normalize_skill_id(payload.skill_id)
    record = next(
        (r for r in doc.skills.installed if _normalize_skill_id(r.id) == normalized),
        None,
    )
    if record is None:
        raise HTTPException(status_code=404, detail=f"skill '{normalized}' is not installed")

    # Remove the on-disk package dir, guarding that the resolved target stays inside
    # the configured local root (never follow it out via symlinks / traversal).
    root = Path(str(getattr(settings, "skill_local_root", "") or skill_local_root())).expanduser().resolve()
    mount_id = normalized.removeprefix("skill.")
    target = (root / mount_id).resolve()
    removed_dir = False
    if target == root or root not in target.parents:
        logger.warning("uninstall: refusing to remove {} (not strictly inside {})", target, root)
    elif target.is_dir():
        try:
            shutil.rmtree(target)
            removed_dir = True
            logger.info("uninstall: removed skill dir {}", target)
        except Exception as exc:
            logger.exception("uninstall: failed to remove {}: {}", target, exc)
            raise HTTPException(status_code=500, detail=f"failed to remove skill files: {exc}") from exc
    else:
        logger.info("uninstall: no on-disk dir at {} (record only)", target)

    result = remove_installed_skill(doc, normalized)
    result["removed_dir"] = removed_dir
    logger.info(
        "uninstall: dropped {} (removed_dir={}, unassigned={})",
        normalized, removed_dir, result["unassigned_agents"],
    )
    await _save_and_reload(doc, state, updated_by=admin.id)
    return JSONResponse({
        "ok": True,
        "result": result,
        "config": (await _runtime_doc(state)).model_dump(mode="json"),
    })
