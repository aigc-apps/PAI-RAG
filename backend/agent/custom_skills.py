from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


# Cache the discovered catalog so build_context (called on every chat turn)
# doesn't re-glob + read_text + yaml-parse every skill manifest per request —
# a real event-loop cost when skills_dir is a NAS mount. Keyed by a cheap
# fingerprint (source paths + each skill dir's mtime), so a skill install /
# remove (which changes a dir mtime) invalidates it naturally on the next call.
_discovery_cache: Dict[str, List[SkillPackage]] = {}


def _discovery_fingerprint(sources: List[Dict[str, Any]]) -> str:
    parts: List[Any] = []
    for source in sources:
        if str(source.get("type") or "local") != "local":
            parts.append(("nonlocal", repr(source)))
            continue
        root = Path(str(source.get("path") or "")).expanduser()
        try:
            root_mtime = root.stat().st_mtime_ns
        except OSError:
            parts.append((str(root), None))
            continue
        children: List[Tuple[str, Optional[int]]] = []
        try:
            with os.scandir(root) as it:  # one round trip; no per-file read/parse
                for entry in it:
                    if entry.is_dir():
                        try:
                            children.append((entry.name, entry.stat().st_mtime_ns))
                        except OSError:
                            children.append((entry.name, None))
        except OSError:
            pass
        parts.append((str(root), root_mtime, tuple(sorted(children))))
    return repr(parts)


def discover_skill_packages(sources: Iterable[Dict[str, Any]]) -> List[SkillPackage]:
    sources = list(sources)
    key = _discovery_fingerprint(sources)
    cached = _discovery_cache.get(key)
    if cached is not None:
        return cached
    packages = _discover_skill_packages_uncached(sources)
    _discovery_cache.clear()  # keep only the current on-disk fingerprint
    _discovery_cache[key] = packages
    return packages


def _discover_skill_packages_uncached(sources: Iterable[Dict[str, Any]]) -> List[SkillPackage]:
    packages: List[SkillPackage] = []
    seen: set[str] = set()
    for source in sources:
        if str(source.get("type") or "local") != "local":
            continue
        root = Path(str(source.get("path") or "")).expanduser()
        if not root.exists() or not root.is_dir():
            continue
        # A skill package is a direct subdirectory with either a skill.yaml
        # (platform manifest) or a SKILL.md (Agent Skills standard). A dir with
        # both (e.g. the demo skill) is loaded once.
        candidate_dirs: List[Path] = []
        candidate_dirs.extend(manifest.parent for manifest in sorted(root.glob("*/skill.yaml")))
        candidate_dirs.extend(skill_md.parent for skill_md in sorted(root.glob("*/SKILL.md")))
        seen_dirs: set[Path] = set()
        for skill_dir in candidate_dirs:
            if skill_dir in seen_dirs:
                continue
            seen_dirs.add(skill_dir)
            package = load_skill_package(skill_dir)
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


def render_skill_catalog(
    *,
    packages: List[SkillPackage],
    enabled_ids: Iterable[str],
) -> str:
    """Always-injected catalog of the agent's enabled skills (name + description
    + capability id) — Level 1 of Agent Skills progressive disclosure. The agent
    always knows *which* skills it has and when to reach for them; the full
    instructions (L2) load on demand via the ``load_skill`` tool and bundled files
    (L3) via ``read_skill_resource``. Nothing here is gated on a lexical query
    match, so a community SKILL.md-only skill with no trigger keywords is still
    visible (the old substring-match gate made such skills invisible)."""
    enabled = {_normalize_skill_id(item) for item in enabled_ids}
    selected = sorted(
        (p for p in packages if _normalize_skill_id(p.capability_id) in enabled),
        key=lambda p: p.capability_id,
    )
    if not selected:
        return ""
    lines = [
        "# Available Skills",
        "You have the following task-specific skills. Each entry below is only a "
        "one-line summary. When a request calls for one, first call the `load_skill` "
        "tool — a tool/function call, never a shell command — with its id to load the "
        "full step-by-step instructions, then follow them; do not attempt the task "
        "from the summary alone. Skills may bundle extra files (templates, references, "
        "scripts); `load_skill` lists them and you read them with the "
        "`read_skill_resource` tool.",
        "",
    ]
    for package in selected:
        description = package.description.strip() or "(no description)"
        lines.append(f"- **{package.name}** (`{package.capability_id}`): {description}")
    return "\n".join(lines)


# Files that are the manifest/instructions themselves — not user-facing bundled
# resources — so they are hidden from the read_skill_resource file listing.
_SKILL_META_FILES = {"skill.yaml", "skill.yml", "SKILL.md", "SKILL.MD"}
_SKILL_SKIP_DIRS = {".git", "__pycache__", ".DS_Store", "node_modules", ".venv"}


def list_skill_files(source_path: str, *, max_files: int = 200) -> List[str]:
    """Relative POSIX paths of the bundled files inside a skill package
    (templates, references, scripts) — everything except the manifest/instruction
    files. Used to advertise Level-3 resources the agent can read on demand."""
    root = Path(source_path)
    if not root.is_dir():
        return []
    out: List[str] = []
    for path in sorted(root.rglob("*")):
        if len(out) >= max_files:
            break
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _SKILL_SKIP_DIRS for part in rel.parts):
            continue
        if rel.name in _SKILL_META_FILES and len(rel.parts) == 1:
            continue
        out.append(rel.as_posix())
    return out


def render_skill_detail(package: SkillPackage, mount_path: Optional[str] = None) -> str:
    """The Level-2 payload returned by the ``load_skill`` tool: the skill's full
    instructions plus a manifest of the bundled files the agent can read via
    ``read_skill_resource``. This is loaded on demand — never preloaded into every
    turn — so context stays lean until a skill is actually needed.

    ``mount_path`` is the skill's read-only mount dir inside the sandbox
    (e.g. ``/mnt/skills/writer``). When known it is stated explicitly so the agent
    reaches its scripts at the right path instead of guessing one from the id."""
    body = package.instructions.strip() or package.description.strip()
    parts = [
        f"# Skill: {package.name} (`{package.capability_id}`)",
        f"Version: {package.version}",
        "",
        body,
    ]
    files = list_skill_files(package.path)
    if files:
        parts.append("")
        if mount_path:
            parts.append(
                f"## Bundled files\nThis skill's files are mounted read-only in the "
                f"sandbox at `{mount_path}` — run its scripts and `ls`/`cat` them there "
                f"with shell / code_interpreter (use that exact path, not one built "
                f"from the skill id). To read a single file host-side instead, use "
                f'`read_skill_resource("{package.capability_id}", "<path>")`:'
            )
        else:
            parts.append(
                "## Bundled files\n"
                f'Read any of these with `read_skill_resource("{package.capability_id}", "<path>")`:'
            )
        parts.extend(f"- {rel}" for rel in files)
    return "\n".join(parts)


def find_enabled_skill_mount(
    skill_mounts: Iterable[Dict[str, Any]], skill_id: str
) -> Optional[Dict[str, Any]]:
    """Resolve a caller-supplied skill id against the enabled mounts on the tool
    scope. Accepts both ``foo`` and ``skill.foo``. Returns the mount dict (with
    ``source_path``) or None when the skill is not enabled for this agent — this
    is the enablement/authorization gate for load_skill / read_skill_resource."""
    target = _normalize_skill_id(str(skill_id).strip())
    for mount in skill_mounts or []:
        if not isinstance(mount, dict):
            continue
        if _normalize_skill_id(str(mount.get("id") or "")) == target:
            return mount
    return None


def load_skill_package(skill_dir: Path) -> Optional[SkillPackage]:
    """Load a skill package from a directory containing either ``skill.yaml``
    (platform manifest, authoritative) or ``SKILL.md`` (Agent Skills standard
    frontmatter). Returns None when the directory is not a skill package.

    Merge rule: ``skill.yaml`` overrides ``SKILL.md`` frontmatter for the fields
    it declares; the SKILL.md body is the instruction text unless skill.yaml's
    ``entry.instructions`` points elsewhere. A SKILL.md-only package synthesizes
    id/name/version/permissions from its frontmatter (spec-aligned), so community
    skills install without a platform manifest.
    """
    skill_yaml = skill_dir / "skill.yaml"
    skill_md = skill_dir / "SKILL.md"
    fm, md_body = _parse_skill_md(skill_md) if skill_md.is_file() else ({}, "")
    if skill_yaml.is_file():
        try:
            data = yaml.safe_load(skill_yaml.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("failed to load skill.yaml {}: {}", skill_yaml, exc)
            return None
        if not isinstance(data, dict):
            return None
        skill_id = str(
            data.get("id") or _slugify(fm.get("name")) or skill_dir.name
        ).strip()
        if not _valid_skill_id(skill_id):
            logger.warning("invalid skill id in {}: {}", skill_yaml, skill_id)
            return None
        entry = data.get("entry") if isinstance(data.get("entry"), dict) else {}
        instructions_file = str(entry.get("instructions") or "SKILL.md")
        instructions = _read_instructions(skill_dir, instructions_file, md_body)
        metadata = fm.get("metadata") if isinstance(fm, dict) else {}
        meta_version = metadata.get("version") if isinstance(metadata, dict) else None
        return SkillPackage(
            id=skill_id,
            name=str(data.get("name") or fm.get("name") or skill_id),
            version=str(data.get("version") or meta_version or "0.0.0"),
            description=str(data.get("description") or fm.get("description") or ""),
            instructions=instructions,
            path=str(skill_dir),
            triggers=data.get("triggers") if isinstance(data.get("triggers"), dict) else {},
            permissions=data.get("permissions") if isinstance(data.get("permissions"), dict) else {},
            resources=[str(item) for item in (data.get("resources") or [])],
            scripts=[str(item) for item in (data.get("scripts") or [])],
        )
    if skill_md.is_file():
        skill_id = _slugify(fm.get("name")) or skill_dir.name
        if not _valid_skill_id(skill_id):
            logger.warning("invalid skill id from SKILL.md {}: {}", skill_md, skill_id)
            return None
        metadata = fm.get("metadata") if isinstance(fm, dict) else {}
        meta_version = metadata.get("version") if isinstance(metadata, dict) else None
        allowed_tools = fm.get("allowed-tools")
        permissions = (
            {"tools": [str(t) for t in allowed_tools]}
            if isinstance(allowed_tools, list)
            else {}
        )
        return SkillPackage(
            id=skill_id,
            name=str(fm.get("name") or skill_id),
            version=str(meta_version or "0.0.0"),
            description=str(fm.get("description") or ""),
            instructions=md_body,
            path=str(skill_dir),
            triggers={},
            permissions=permissions,
            resources=[],
            scripts=[],
        )
    return None


def _parse_skill_md(path: Path) -> Tuple[Dict[str, Any], str]:
    """Split a SKILL.md into (frontmatter_dict, body). The whole file is body
    when there is no leading ``---`` frontmatter block."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        return {}, text
    if not isinstance(fm, dict):
        return {}, text
    return fm, parts[2].lstrip("\n")


def _read_instructions(skill_dir: Path, instructions_file: str, md_body: str) -> str:
    """Read the instruction text for a skill.yaml-declared entry. When the entry
    is SKILL.md, reuse the already-parsed body; otherwise read the named file."""
    if instructions_file in ("SKILL.md", "SKILL.MD"):
        return md_body
    path = skill_dir / instructions_file
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def _slugify(value: Any) -> str:
    """Lowercase, runs of non-[a-z0-9] -> single '-', strip edges. Maps an
    Agent Skills frontmatter ``name`` to a safe mount/capability id."""
    if not value:
        return ""
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower())
    return text.strip("-")


def _valid_skill_id(skill_id: str) -> bool:
    return bool(skill_id) and "/" not in skill_id and "\\" not in skill_id and ".." not in skill_id


def _normalize_skill_id(skill_id: str) -> str:
    return skill_id if skill_id.startswith("skill.") else f"skill.{skill_id}"


def _skill_nas_mount(package: SkillPackage, mount_path: str, nas_cfg: Dict[str, Any]) -> Dict[str, Any]:
    server_addr = nas_cfg.get("serverAddr") or nas_cfg.get("server_addr")
    if not server_addr:
        return {}
    prefix = str(nas_cfg.get("remotePathPrefix") or nas_cfg.get("remote_path_prefix") or "skills").strip("/")
    # Remote path on the NAS filesystem, e.g. "/skills/writer". Unversioned: it
    # must match the dir install_skill writes (root/<mount_id>, no @version), so
    # the sandbox's read-only mount lands on the freshly installed files.
    remote_path = f"/{prefix}/{package.mount_id}".replace("//", "/")
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
