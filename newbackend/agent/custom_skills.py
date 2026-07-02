from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from loguru import logger


@dataclass
class SkillPackage:
    id: str
    name: str
    version: str = "0.0.0"
    description: str = ""
    instructions: str = ""
    path: str = ""
    triggers: Dict[str, Any] = field(default_factory=dict)
    permissions: Dict[str, Any] = field(default_factory=dict)
    resources: List[str] = field(default_factory=list)
    scripts: List[str] = field(default_factory=list)

    @property
    def capability_id(self) -> str:
        return self.id if self.id.startswith("skill.") else f"skill.{self.id}"

    @property
    def mount_id(self) -> str:
        return self.id.removeprefix("skill.")


@dataclass
class SkillMount:
    id: str
    version: str
    source_path: str
    mount_path: str
    read_only: bool = True
    nas: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "source_path": self.source_path,
            "mount_path": self.mount_path,
            "read_only": self.read_only,
            "nas": self.nas,
        }


def discover_skill_packages(sources: Iterable[Dict[str, Any]]) -> List[SkillPackage]:
    packages: List[SkillPackage] = []
    seen: set[str] = set()
    for source in sources:
        if str(source.get("type") or "local") != "local":
            continue
        root = Path(str(source.get("path") or "")).expanduser()
        if not root.exists() or not root.is_dir():
            continue
        for manifest in sorted(root.glob("*/skill.yaml")):
            package = _load_skill_package(manifest)
            if package is None or package.id in seen:
                continue
            seen.add(package.id)
            packages.append(package)
    return packages


def skill_sources(skill_config: Any) -> List[Dict[str, Any]]:
    root = getattr(skill_config, "root", "") or ""
    return [{"type": "local", "path": root}] if root else []


def resolve_skill_mounts(
    *,
    packages: List[SkillPackage],
    enabled_ids: Iterable[str],
    skill_config: Any,
) -> List[SkillMount]:
    enabled = {_normalize_skill_id(item) for item in enabled_ids}
    mount_cfg = getattr(skill_config, "mount", {}) or {}
    mount_root = str(mount_cfg.get("mount_root") or "/mnt/skills").rstrip("/")
    nas_cfg = mount_cfg.get("nas") if isinstance(mount_cfg.get("nas"), dict) else {}
    mounts: List[SkillMount] = []
    for package in packages:
        if _normalize_skill_id(package.capability_id) not in enabled:
            continue
        mount_path = f"{mount_root}/{package.mount_id}"
        nas = _skill_nas_mount(package, mount_path, nas_cfg)
        mounts.append(
            SkillMount(
                id=package.capability_id,
                version=package.version,
                source_path=package.path,
                mount_path=mount_path,
                read_only=True,
                nas=nas,
            )
        )
    return mounts


def skill_mount_fingerprint(mounts: List[SkillMount]) -> str:
    if not mounts:
        return "none"
    payload = [
        {
            "id": mount.id,
            "version": mount.version,
            "source_path": mount.source_path,
            "mount_path": mount.mount_path,
            "nas": mount.nas,
        }
        for mount in sorted(mounts, key=lambda item: item.id)
    ]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def render_skill_instructions(
    *,
    packages: List[SkillPackage],
    enabled_ids: Iterable[str],
    query: str,
    max_skills: int = 3,
) -> str:
    enabled = {_normalize_skill_id(item) for item in enabled_ids}
    selected = [
        package
        for package in packages
        if _normalize_skill_id(package.capability_id) in enabled
        and _skill_matches(package, query)
    ][:max_skills]
    if not selected:
        return ""
    blocks = []
    for package in selected:
        body = package.instructions.strip()
        if not body:
            body = package.description.strip()
        if not body:
            continue
        blocks.append(
            f"## {package.name} ({package.capability_id})\n"
            f"Version: {package.version}\n"
            f"Source: {package.path}\n\n"
            f"{body}"
        )
    if not blocks:
        return ""
    return "# Active Skills\nUse these task-specific skill instructions when relevant.\n\n" + "\n\n".join(blocks)


def _load_skill_package(manifest: Path) -> Optional[SkillPackage]:
    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return None
        skill_id = str(data.get("id") or manifest.parent.name).strip()
        if not skill_id:
            return None
        entry = (data.get("entry") or {}).get("instructions") if isinstance(data.get("entry"), dict) else None
        instruction_file = manifest.parent / str(entry or "SKILL.md")
        instructions = ""
        if instruction_file.exists() and instruction_file.is_file():
            instructions = instruction_file.read_text(encoding="utf-8")
        return SkillPackage(
            id=skill_id,
            name=str(data.get("name") or skill_id),
            version=str(data.get("version") or "0.0.0"),
            description=str(data.get("description") or ""),
            instructions=instructions,
            path=str(manifest.parent),
            triggers=data.get("triggers") if isinstance(data.get("triggers"), dict) else {},
            permissions=data.get("permissions") if isinstance(data.get("permissions"), dict) else {},
            resources=[str(item) for item in (data.get("resources") or [])],
            scripts=[str(item) for item in (data.get("scripts") or [])],
        )
    except Exception as exc:
        logger.warning("failed to load skill package {}: {}", manifest, exc)
        return None


def _skill_matches(package: SkillPackage, query: str) -> bool:
    query_l = query.lower()
    keywords = package.triggers.get("keywords") or []
    if isinstance(keywords, list):
        for keyword in keywords:
            if str(keyword).lower() in query_l:
                return True
    intents = package.triggers.get("intents") or []
    if isinstance(intents, list):
        for intent in intents:
            if str(intent).replace("_", " ").lower() in query_l:
                return True
    text = f"{package.name} {package.description}".lower()
    return any(part and part in query_l for part in text.split())


def _normalize_skill_id(skill_id: str) -> str:
    return skill_id if skill_id.startswith("skill.") else f"skill.{skill_id}"


def _skill_nas_mount(package: SkillPackage, mount_path: str, nas_cfg: Dict[str, Any]) -> Dict[str, Any]:
    server_addr = nas_cfg.get("serverAddr") or nas_cfg.get("server_addr")
    if not server_addr:
        return {}
    prefix = str(nas_cfg.get("remotePathPrefix") or nas_cfg.get("remote_path_prefix") or "skills").strip("/")
    # Remote path on the NAS filesystem, e.g. "/skills/writer@1.0.0".
    remote_path = f"/{prefix}/{package.mount_id}@{package.version}".replace("//", "/")
    # serverAddr is the NAS mount point with the remote path appended, e.g.
    # "xxxx.nas.aliyuncs.com:/skills/writer@1.0.0".
    if server_addr.endswith(":/"):
        full_server_addr = f"{server_addr}{remote_path.lstrip('/')}"
    elif ":/" in server_addr:
        full_server_addr = f"{server_addr.rstrip('/')}/{remote_path.lstrip('/')}"
    else:
        full_server_addr = f"{server_addr}:{remote_path}"
    return {
        "serverAddr": full_server_addr,
        "remotePath": remote_path,
        "mountDir": mount_path,
        "readOnly": bool(nas_cfg.get("readOnly", nas_cfg.get("read_only", True))),
    }
