from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from agent.custom_skills import SkillPackage
from agent.tools.base import Tool


def make_install_skill_tool(settings, agent_config) -> Tool:
    skill_config = getattr(agent_config, "skills", None)
    root = Path(str(getattr(skill_config, "root", "./data/skills"))).expanduser()
    install_config = dict(getattr(skill_config, "install", {}) or {})
    upload_root = Path(str(install_config.get("upload_root") or "./data/skill-uploads")).expanduser()
    app_env = str(getattr(settings, "app_env", "development") or "development").lower()

    async def _install_skill(
        source: Dict[str, Any],
        enable_for_agent: Optional[str] = None,
        enable_after_build: bool = False,
        overwrite: bool = False,
    ) -> str:
        result = await asyncio.to_thread(
            _install_skill_sync,
            source,
            root,
            install_config,
            app_env,
            bool(overwrite),
            enable_for_agent,
            bool(enable_after_build),
            upload_root,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    return Tool(
        name="install_skill",
        description=(
            "Admin-only control-plane tool for installing a skill package from "
            "URL or Git. URL/Git installs are disabled in production."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source": {
                    "type": "object",
                    "description": "Skill source: {type:url,url,checksum} or {type:git,url,ref,path}.",
                    "properties": {
                        "type": {"type": "string", "enum": ["zip_upload", "url", "git"]},
                        "upload_id": {"type": "string"},
                        "url": {"type": "string"},
                        "checksum": {"type": "string"},
                        "ref": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["type"],
                    "additionalProperties": True,
                },
                "enable_for_agent": {
                    "type": "string",
                    "description": "Optional agent id to enable after dependency build.",
                },
                "enable_after_build": {
                    "type": "boolean",
                    "description": "Requested final intent; first version reports it but does not mutate config.",
                },
                "overwrite": {"type": "boolean"},
            },
            "required": ["source"],
            "additionalProperties": False,
        },
        fn=_install_skill,
        permission="admin",
    )


def _install_skill_sync(
    source: Dict[str, Any],
    root: Path,
    install_config: Dict[str, Any],
    app_env: str,
    overwrite: bool,
    enable_for_agent: Optional[str],
    enable_after_build: bool,
    upload_root: Optional[Path] = None,
) -> Dict[str, Any]:
    source_type = str(source.get("type") or "")
    _validate_source_allowed(source_type, install_config, app_env)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="skill-install-") as tmp:
        tmp_path = Path(tmp)
        if source_type == "url":
            package_dir, source_meta = _prepare_url_source(source, tmp_path, install_config)
        elif source_type == "git":
            package_dir, source_meta = _prepare_git_source(source, tmp_path)
        elif source_type == "zip_upload":
            package_dir, source_meta = _prepare_zip_upload_source(source, tmp_path, upload_root)
        else:
            raise ValueError(f"unsupported skill source type: {source_type}")

        package = _read_skill_package(package_dir)
        target_dir = root / package.mount_id
        if target_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"skill '{package.capability_id}' already exists at {target_dir}; set overwrite=true"
                )
            shutil.rmtree(target_dir)
        shutil.copytree(package_dir, target_dir, symlinks=False)
        dependencies = _dependency_summary(target_dir)
        status = "building_dependencies" if dependencies["has_dependencies"] else "ready"
        return {
            "id": package.capability_id,
            "name": package.name,
            "version": package.version,
            "path": str(target_dir),
            "status": status,
            "dependency_status": "pending_build" if dependencies["has_dependencies"] else "none",
            "dependencies": dependencies,
            "source": source_meta,
            "enable_for_agent": enable_for_agent,
            "enable_after_build": enable_after_build,
        }


def _validate_source_allowed(source_type: str, install_config: Dict[str, Any], app_env: str) -> None:
    allowed = install_config.get("allow_sources") or ["zip_upload", "url", "git"]
    production_allowed = install_config.get("production_allow_sources") or ["zip_upload"]
    active_allowed = production_allowed if app_env == "production" else allowed
    if source_type not in active_allowed:
        raise PermissionError(f"skill install source '{source_type}' is disabled in {app_env}")


def _prepare_url_source(
    source: Dict[str, Any], tmp_path: Path, install_config: Dict[str, Any]
) -> tuple[Path, Dict[str, Any]]:
    url = str(source.get("url") or "")
    if not url.startswith("https://"):
        raise ValueError("url skill install requires an https URL")
    archive = tmp_path / "skill.zip"
    max_bytes = int(install_config.get("max_download_mb") or 50) * 1024 * 1024
    digest = _download(url, archive, max_bytes)
    checksum = str(source.get("checksum") or "")
    if checksum:
        expected = checksum.removeprefix("sha256:")
        if digest != expected:
            raise ValueError("downloaded skill checksum mismatch")
    extract_dir = tmp_path / "extract"
    _extract_zip_safe(archive, extract_dir)
    return _find_skill_dir(extract_dir), {
        "type": "url",
        "url": url,
        "sha256": digest,
    }


def _prepare_zip_upload_source(
    source: Dict[str, Any], tmp_path: Path, upload_root: Optional[Path]
) -> tuple[Path, Dict[str, Any]]:
    upload_id = str(source.get("upload_id") or "")
    if not upload_id:
        raise ValueError("zip_upload source requires upload_id")
    if "/" in upload_id or "\\" in upload_id or ".." in upload_id:
        raise ValueError("upload_id must be a safe id")
    if upload_root is None:
        raise ValueError("zip_upload install requires upload_root")
    archive = upload_root / f"{upload_id}.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"uploaded skill archive not found: {upload_id}")
    digest = _file_sha256(archive)
    extract_dir = tmp_path / "extract"
    _extract_zip_safe(archive, extract_dir)
    return _find_skill_dir(extract_dir), {
        "type": "zip_upload",
        "upload_id": upload_id,
        "sha256": digest,
    }


def _prepare_git_source(source: Dict[str, Any], tmp_path: Path) -> tuple[Path, Dict[str, Any]]:
    url = str(source.get("url") or "")
    if not url.startswith("https://"):
        raise ValueError("git skill install requires an https URL")
    repo_dir = tmp_path / "repo"
    cmd = ["git", "clone", "--depth", "1"]
    ref = str(source.get("ref") or "")
    if ref:
        cmd.extend(["--branch", ref])
    cmd.extend([url, str(repo_dir)])
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    commit = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    package_path = _safe_relative_path(str(source.get("path") or "."))
    return _find_skill_dir(repo_dir / package_path), {
        "type": "git",
        "url": url,
        "ref": ref or None,
        "commit": commit,
        "path": str(package_path),
    }


def _download(url: str, target: Path, max_bytes: int) -> str:
    digest = hashlib.sha256()
    downloaded = 0
    with urllib.request.urlopen(url, timeout=30) as response, target.open("wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            downloaded += len(chunk)
            if downloaded > max_bytes:
                raise ValueError("skill archive exceeds configured max_download_mb")
            digest.update(chunk)
            fh.write(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_zip_safe(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = member.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe zip path: {name}")
            if member.is_dir():
                continue
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)


def _find_skill_dir(root: Path) -> Path:
    if (root / "skill.yaml").is_file():
        return root
    manifests = sorted(root.glob("*/skill.yaml"))
    if len(manifests) == 1:
        return manifests[0].parent
    if not manifests:
        raise ValueError("skill package missing skill.yaml")
    raise ValueError("archive/repo path contains multiple skill packages; specify source.path")


def _read_skill_package(path: Path) -> SkillPackage:
    manifest = path / "skill.yaml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("skill.yaml must be a mapping")
    skill_id = str(data.get("id") or path.name).strip()
    if not skill_id:
        raise ValueError("skill.yaml missing id")
    if "/" in skill_id or "\\" in skill_id or ".." in skill_id:
        raise ValueError("skill id must be a simple package id")
    entry = data.get("entry") if isinstance(data.get("entry"), dict) else {}
    instructions = str(entry.get("instructions") or "SKILL.md")
    if not (path / instructions).is_file():
        raise ValueError(f"skill instructions file not found: {instructions}")
    return SkillPackage(
        id=skill_id,
        name=str(data.get("name") or skill_id),
        version=str(data.get("version") or "0.0.0"),
        description=str(data.get("description") or ""),
        path=str(path),
    )


def _dependency_summary(path: Path) -> Dict[str, Any]:
    requirements = path / "requirements.txt"
    package_json = path / "package.json"
    runtime = {}
    manifest = path / "skill.yaml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict) and isinstance(data.get("runtime"), dict):
        runtime = data["runtime"]
    return {
        "has_dependencies": requirements.is_file() or package_json.is_file() or bool(runtime),
        "python": {"requirements": "requirements.txt"} if requirements.is_file() else {},
        "node": {"package": "package.json"} if package_json.is_file() else {},
        "runtime": runtime,
    }


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("source.path must be a safe relative path")
    return path
