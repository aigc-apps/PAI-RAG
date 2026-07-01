from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from agent.custom_skills import discover_skill_packages, skill_sources


Permission = Literal["disabled", "ask", "auto", "admin"]
CapabilityStatus = Literal["ready", "missing_config", "error", "disabled"]
ProviderStatus = Literal["untested", "healthy", "missing_config", "error"]


class ProviderConfig(BaseModel):
    id: str
    type: Literal["llm", "search", "embedding", "rerank", "vectordb", "sandbox"]
    name: str
    status: ProviderStatus = "untested"
    settings: Dict[str, Any] = Field(default_factory=dict)
    secret_configured: bool = False
    error: Optional[str] = None
    used_by: List[str] = Field(default_factory=list)


class CapabilityConfig(BaseModel):
    id: str
    kind: Literal["core_tool", "skill"]
    name: str
    description: str = ""
    enabled: bool = False
    permission: Permission = "disabled"
    status: CapabilityStatus = "disabled"
    dependencies: List[str] = Field(default_factory=list)
    provider_refs: List[str] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class SetupConfig(BaseModel):
    completed: bool = False
    completed_at: Optional[str] = None
    mode: Optional[Literal["local_first", "cloud_enhanced", "developer"]] = None
    skipped_steps: List[str] = Field(default_factory=list)


class AgentToolsConfig(BaseModel):
    include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)


class AgentSkillsConfig(BaseModel):
    enabled: List[str] = Field(default_factory=list)


class SkillLibraryConfig(BaseModel):
    root: str = "./data/skills"
    mount: Dict[str, Any] = Field(default_factory=lambda: {"mount_root": "/mnt/skills"})
    install: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "admin_only": True,
            "allow_sources": ["zip_upload", "url", "git"],
            "production_allow_sources": ["zip_upload"],
            "require_review": True,
            "max_download_mb": 50,
            "upload_root": "./data/skill-uploads",
        }
    )
    dependencies: Dict[str, Any] = Field(
        default_factory=lambda: {
            "mode": "prebuilt",
            "env_root": "/mnt/skill-envs",
            "admin_build_enabled": True,
            "runtime_install_enabled": False,
            "build_timeout_seconds": 600,
        }
    )
    config: Dict[str, Any] = Field(default_factory=dict)
    installed: List[Dict[str, Any]] = Field(default_factory=list)


class AgentProfile(BaseModel):
    id: str
    name: str
    description: str = ""
    model: str = ""
    instructions: str = ""
    tools: AgentToolsConfig = Field(default_factory=AgentToolsConfig)
    skills: AgentSkillsConfig = Field(default_factory=AgentSkillsConfig)
    settings: Dict[str, Any] = Field(default_factory=dict)


class AgentConfigDocument(BaseModel):
    setup: SetupConfig = Field(default_factory=SetupConfig)
    models: Dict[str, Any] = Field(default_factory=dict)
    skills: SkillLibraryConfig = Field(default_factory=SkillLibraryConfig)
    default_agent: str = "main"
    agents: List[AgentProfile] = Field(default_factory=list)
    providers: List[ProviderConfig] = Field(default_factory=list)
    capabilities: List[CapabilityConfig] = Field(default_factory=list)


DEFAULT_DOCUMENT = AgentConfigDocument(
    models={
        "default_model": "openai/gpt-4o-mini",
        "providers": [
            {
                "name": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "models": [
                    {
                        "id": "gpt-4o-mini",
                        "context_window": 128000,
                        "max_output_tokens": 16384,
                        "supports_tools": True,
                    }
                ],
            }
        ],
    },
    skills=SkillLibraryConfig(root="./data/skills", mount={"mount_root": "/mnt/skills"}),
    default_agent="main",
    agents=[
        AgentProfile(
            id="main",
            name="Main",
            description="General-purpose assistant using the default model and safe core tools.",
            model="",
            instructions="",
            tools=AgentToolsConfig(
                include=[
                    "current_datetime",
                    "web_fetch",
                    "web_search",
                    "knowledge_search",
                ],
                exclude=["code_sandbox"],
            ),
            skills=AgentSkillsConfig(
                enabled=["skill.writing", "skill.knowledge_qa"]
            ),
            settings={"max_steps": 20},
        )
    ],
    providers=[
        ProviderConfig(
            id="llm.default",
            type="llm",
            name="Default model provider",
            used_by=["model"],
        ),
        ProviderConfig(
            id="search.default",
            type="search",
            name="Search provider",
            settings={
                "provider": "none",
                "endpoint": "",
                "api_key_env": "",
                "max_results": 5,
            },
            used_by=["search"],
        ),
        ProviderConfig(
            id="embedding.default",
            type="embedding",
            name="Embedding provider",
            used_by=["knowledge"],
        ),
        ProviderConfig(
            id="rerank.default",
            type="rerank",
            name="Rerank provider",
            used_by=["knowledge"],
        ),
        ProviderConfig(
            id="vectordb.default",
            type="vectordb",
            name="Vector DB provider",
            used_by=["knowledge"],
        ),
        ProviderConfig(
            id="sandbox.default",
            type="sandbox",
            name="Sandbox runtime",
            settings={
                "provider": "agentrun_rest",
                "endpoint": "",
                "api_key_env": "AGENTRUN_SANDBOX_API_KEY",
                "api_key_header": "X-API-Key",
                "template_name": "",
                "template_type": "CodeInterpreter",
                "region": "cn-hangzhou",
                "access_key_id_env": "AGENTRUN_ACCESS_KEY_ID",
                "access_key_secret_env": "AGENTRUN_ACCESS_KEY_SECRET",
                "account_id_env": "AGENTRUN_ACCOUNT_ID",
                "isolation_scope": "conversation",
                "idle_timeout_seconds": 600,
                "session_idle_seconds": 600,
                "timeout_seconds": 30,
                "cwd": "/home/user",
                "create_path": "/sandboxes",
                "execute_path": "/sandboxes/{sandbox_id}/contexts/execute",
                "stop_path": "/sandboxes/{sandbox_id}/stop",
                "oss_mount_config": {
                    "mount_points": []
                },
                "nas_config": {
                    "mount_points": []
                },
            },
            used_by=["sandbox"],
        ),
    ],
    capabilities=[
        CapabilityConfig(
            id="search",
            kind="core_tool",
            name="Search",
            description="Let the agent retrieve current information from the web.",
            enabled=False,
            permission="auto",
            provider_refs=["search.default"],
        ),
        CapabilityConfig(
            id="knowledge",
            kind="core_tool",
            name="Knowledge Base",
            description="Retrieve from local documents, with optional cloud-enhanced retrieval.",
            enabled=True,
            permission="auto",
            status="ready",
            provider_refs=["embedding.default", "rerank.default", "vectordb.default"],
            settings={"mode": "local"},
        ),
        CapabilityConfig(
            id="sandbox",
            kind="core_tool",
            name="Sandbox",
            description="Run scripts in a constrained local or cloud runtime.",
            enabled=False,
            permission="disabled",
            provider_refs=["sandbox.default"],
            settings={"runtime": "local", "network": False},
        ),
        CapabilityConfig(
            id="install_skill",
            kind="core_tool",
            name="Install Skill",
            description="Admin-only tool for installing skill packages from zip upload, URL, or Git.",
            enabled=False,
            permission="admin",
            status="disabled",
            settings={"control_plane": True},
        ),
        CapabilityConfig(
            id="skill.knowledge_qa",
            kind="skill",
            name="Knowledge QA",
            description="Answer questions grounded in configured knowledge collections.",
            enabled=True,
            permission="auto",
            status="ready",
            dependencies=["knowledge"],
        ),
        CapabilityConfig(
            id="skill.data_analysis",
            kind="skill",
            name="Data Analysis",
            description="Analyze files or data by executing scripts in a sandbox.",
            enabled=False,
            permission="ask",
            dependencies=["sandbox"],
        ),
        CapabilityConfig(
            id="skill.writing",
            kind="skill",
            name="Writing Assistant",
            description="Draft and revise structured content without extra tools.",
            enabled=True,
            permission="auto",
            status="ready",
        ),
    ],
)


def _merge_default(raw: Dict[str, Any]) -> AgentConfigDocument:
    base = DEFAULT_DOCUMENT.model_dump(mode="json")
    merged = deepcopy(base)
    for key in ("setup",):
        if isinstance(raw.get(key), dict):
            merged[key].update(raw[key])
    if isinstance(raw.get("default_agent"), str):
        merged["default_agent"] = raw["default_agent"]
    if isinstance(raw.get("models"), dict):
        merged["models"] = raw["models"]
    if isinstance(raw.get("skills"), dict):
        merged["skills"].update(raw["skills"])

    for collection in ("agents", "providers", "capabilities"):
        by_id = {item["id"]: item for item in merged[collection]}
        for item in raw.get(collection, []) or []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            if item["id"] in by_id:
                by_id[item["id"]].update(item)
            else:
                merged[collection].append(item)
    return AgentConfigDocument(**merged)


def load_agent_config(path: str) -> AgentConfigDocument:
    p = Path(path)
    if not p.exists():
        return DEFAULT_DOCUMENT.model_copy(deep=True)
    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() in {".yaml", ".yml"}:
            return _merge_default(yaml.safe_load(f) or {})
        return _merge_default(json.load(f) or {})


def save_agent_config(path: str, doc: AgentConfigDocument) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        data = doc.model_dump(mode="json")
        if p.suffix.lower() in {".yaml", ".yml"}:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, p)


def _configured_secret(settings_dict: Dict[str, Any]) -> bool:
    for direct_key, env_key in (
        ("api_key", "api_key_env"),
        ("access_key_secret", "access_key_secret_env"),
        ("access_key_id", "access_key_id_env"),
    ):
        direct = str(settings_dict.get(direct_key) or "")
        env_name = str(settings_dict.get(env_key) or "")
        if direct or (env_name and os.environ.get(env_name, "")):
            return True
    return False


def _setting_configured(settings_dict: Dict[str, Any], direct_key: str, env_key: str) -> bool:
    direct = str(settings_dict.get(direct_key) or "")
    env_name = str(settings_dict.get(env_key) or "")
    return bool(direct or (env_name and os.environ.get(env_name, "")))


def mask_secrets(doc: AgentConfigDocument) -> AgentConfigDocument:
    out = doc.model_copy(deep=True)
    for provider in out.providers:
        for key in ("api_key", "access_key_id", "access_key_secret", "security_token"):
            if provider.settings.get(key):
                provider.settings[key] = "********"
    return out


def apply_runtime_status(doc: AgentConfigDocument, settings, router) -> AgentConfigDocument:
    out = doc.model_copy(deep=True)
    _merge_discovered_skills(out)
    providers = {p.id: p for p in out.providers}
    caps = {c.id: c for c in out.capabilities}

    llm = providers.get("llm.default")
    if llm is not None:
        model_ready = False
        if router is not None and router.default_model_id:
            try:
                cfg = router.get_config(router.default_model_id)
                model_ready = bool(cfg.resolve_key() or not cfg.api_key_env)
                llm.settings.update({
                    "default_model": router.default_model_id,
                    "provider": cfg.provider,
                    "base_url": cfg.base_url,
                })
                llm.secret_configured = bool(cfg.resolve_key())
            except Exception as exc:
                llm.error = str(exc)
        else:
            model_ready = bool(getattr(settings, "openai_api_key", ""))
            llm.settings.update({"default_model": getattr(settings, "default_model", "")})
            llm.secret_configured = model_ready
        llm.status = "healthy" if model_ready else "missing_config"
        for agent in out.agents:
            if not agent.model:
                agent.model = str(llm.settings.get("default_model") or "")

    search_provider = providers.get("search.default")
    search_cap = caps.get("search")
    if search_provider is not None:
        provider_name = (
            search_provider.settings.get("provider")
            or getattr(settings, "search_provider", "none")
            or "none"
        )
        if provider_name == "none" and getattr(settings, "search_provider", "none") != "none":
            provider_name = getattr(settings, "search_provider")
        if not search_provider.settings.get("api_key") and getattr(settings, "search_api_key", ""):
            search_provider.settings["api_key"] = getattr(settings, "search_api_key")
        if not search_provider.settings.get("endpoint") and getattr(settings, "search_endpoint", ""):
            search_provider.settings["endpoint"] = getattr(settings, "search_endpoint")
        search_provider.name = provider_name if provider_name != "none" else "Search provider"
        search_provider.secret_configured = _configured_secret(search_provider.settings)
        search_provider.settings.update({
            "provider": provider_name,
        })
        search_provider.status = (
            "healthy"
            if provider_name != "none" and search_provider.secret_configured
            else "missing_config"
        )
        if search_cap is not None and search_cap.enabled:
            search_cap.status = "ready" if search_provider.status == "healthy" else "missing_config"
            search_cap.error = None if search_cap.status == "ready" else "Search provider is not configured"

    knowledge = caps.get("knowledge")
    if knowledge is not None:
        if knowledge.settings.get("mode", "local") == "local":
            knowledge.status = "ready" if knowledge.enabled else "disabled"
        elif knowledge.enabled:
            missing = [
                ref for ref in knowledge.provider_refs
                if providers.get(ref) is None or providers[ref].status != "healthy"
            ]
            knowledge.status = "missing_config" if missing else "ready"
            knowledge.error = f"Missing providers: {', '.join(missing)}" if missing else None

    sandbox_provider = providers.get("sandbox.default")
    sandbox = caps.get("sandbox")
    if sandbox_provider is not None:
        sandbox_settings = sandbox_provider.settings or {}
        sandbox_provider_name = sandbox_settings.get("provider")
        configured = False
        if sandbox_provider_name == "agentrun_rest":
            configured = bool(
                sandbox_settings.get("endpoint")
                and sandbox_settings.get("template_name")
                and _setting_configured(sandbox_settings, "account_id", "account_id_env")
            )
        elif sandbox_provider_name == "agentrun":
            configured = (
                bool(sandbox_settings.get("template_name"))
                and _setting_configured(sandbox_settings, "access_key_id", "access_key_id_env")
                and _setting_configured(sandbox_settings, "access_key_secret", "access_key_secret_env")
                and _setting_configured(sandbox_settings, "account_id", "account_id_env")
            )
        sandbox_provider.status = (
            "healthy" if sandbox and sandbox.enabled and configured
            else "missing_config" if sandbox and sandbox.enabled
            else "untested"
        )
        sandbox_provider.secret_configured = _configured_secret(sandbox_settings)
        sandbox_provider.error = None if sandbox_provider.status != "missing_config" else (
            "Configure sandbox provider, endpoint/template, and credentials when required"
        )
    if sandbox is not None:
        sandbox.status = (
            "ready"
            if sandbox.enabled and sandbox_provider and sandbox_provider.status == "healthy"
            else "missing_config" if sandbox.enabled
            else "disabled"
        )
        sandbox.error = None if sandbox.status == "ready" else (
            sandbox_provider.error if sandbox_provider else "Sandbox provider is not configured"
        )

    for cap in out.capabilities:
        if cap.kind != "skill":
            if not cap.enabled:
                cap.status = "disabled"
            elif cap.id == "install_skill":
                cap.status = "ready"
                cap.error = None
            continue
        if not cap.enabled:
            cap.status = "disabled"
            continue
        missing = [dep for dep in cap.dependencies if caps.get(dep) is None or caps[dep].status != "ready"]
        cap.status = "missing_config" if missing else "ready"
        cap.error = f"Requires: {', '.join(missing)}" if missing else None

    return out


def _merge_discovered_skills(doc: AgentConfigDocument) -> None:
    existing = {cap.id: cap for cap in doc.capabilities}
    for package in discover_skill_packages(skill_sources(doc.skills)):
        cap_id = package.capability_id
        current = existing.get(cap_id)
        if current is None:
            doc.capabilities.append(
                CapabilityConfig(
                    id=cap_id,
                    kind="skill",
                    name=package.name,
                    description=package.description,
                    enabled=True,
                    permission="auto",
                    status="ready",
                    dependencies=_skill_tool_dependencies(package.permissions),
                    settings={
                        "source": "local",
                        "path": package.path,
                        "version": package.version,
                        "resources": package.resources,
                        "scripts": package.scripts,
                        "triggers": package.triggers,
                    },
                )
            )
            existing[cap_id] = doc.capabilities[-1]
        else:
            current.name = current.name or package.name
            current.description = current.description or package.description
            current.settings.update({
                "source": "local",
                "path": package.path,
                "version": package.version,
                "resources": package.resources,
                "scripts": package.scripts,
                "triggers": package.triggers,
            })


def _skill_tool_dependencies(permissions: Dict[str, Any]) -> List[str]:
    mapped = {
        "web_search": "search",
        "knowledge_search": "knowledge",
        "code_sandbox": "sandbox",
    }
    tools = permissions.get("tools") or []
    if not isinstance(tools, list):
        return []
    return [mapped.get(str(tool), str(tool)) for tool in tools]
