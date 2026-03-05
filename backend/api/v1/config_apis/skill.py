"""Skills Configuration API - CRUD operations for skill management.

Supports:
- List all skills (with pagination)
- Get a single skill by ID
- Install a skill by uploading .md content or file
- Update skill metadata
- Toggle skill enabled/disabled
- Delete a skill
- Install from uploaded .md file
"""

import os
import json
import shutil
import traceback
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form
from sqlmodel.ext.asyncio.session import AsyncSession
from common.chat.response_model import PagedResult, ResponseModel, success_response
from db.models.skill import SkillRead, SkillCreate, SkillUpdate
from db.db_context import get_db_session
from service.tool.skill_service import SkillService
from service.injection import get_skill_service, get_tenant_id
from api.api_exception import ApiException
from skills.loader import (
    load_skill_from_file,
    load_skill_from_directory,
    parse_frontmatter,
    parsed_skill_to_db_fields,
)
from loguru import logger
from common.i18n import i18n
from typing import Optional

skill_router = APIRouter()


@skill_router.get("", response_model=ResponseModel[PagedResult])
async def list_skills(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, le=1000),
    enabled_only: bool = Query(default=False),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    skill_service: SkillService = Depends(get_skill_service),
):
    """List all installed skills."""
    try:
        result = await skill_service.list_skills(
            tenant_id=tenant_id,
            page=page,
            size=size,
            enabled_only=enabled_only,
        )
        return success_response(data=result, message="Skills listed successfully.")
    except Exception as e:
        logger.error(f"Failed to list skills: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to list skills: {e}")


@skill_router.get("/{skill_id}", response_model=SkillRead)
async def get_skill(
    skill_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    skill_service: SkillService = Depends(get_skill_service),
):
    """Get a single skill by ID."""
    try:
        skill = await skill_service.get_skill(skill_id=skill_id, tenant_id=tenant_id)
        if not skill:
            raise ApiException(code=404, message=f"Skill '{skill_id}' not found.")
        return success_response(data=skill, message="Skill retrieved successfully.")
    except ApiException:
        raise
    except Exception as e:
        logger.error(f"Failed to get skill: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to get skill: {e}")


@skill_router.post("", response_model=SkillRead)
async def create_skill(
    skill_data: SkillCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    skill_service: SkillService = Depends(get_skill_service),
):
    """Create a skill from JSON body (manual creation)."""
    try:
        skill = await skill_service.create_skill(skill_data=skill_data, tenant_id=tenant_id)
        return success_response(data=skill, message="Skill created successfully.")
    except ValueError as e:
        logger.error(f"Failed to create skill: {traceback.format_exc()}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to create skill: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to create skill: {e}")


@skill_router.post("/install", response_model=SkillRead)
async def install_skill(
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    skill_service: SkillService = Depends(get_skill_service),
):
    """Install a skill by uploading a .md file.

    Accepts a single .md file. The file is parsed for YAML frontmatter
    to extract skill metadata, and the body becomes the skill instructions.
    """
    if not file.filename:
        raise ApiException(code=400, message="File is required.")

    try:
        # Read file content
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8")

        # Check if it's a .md file
        if not file.filename.lower().endswith(".md"):
            raise ApiException(code=400, message="Only .md files are supported for skill installation.")

        # Parse the .md content using the loader
        fm, body = parse_frontmatter(content_str)

        # Build skill create data
        name = fm.pop("name", None) or Path(file.filename).stem
        description = fm.pop("description", "")
        tools = fm.pop("tools", [])
        env = fm.pop("env", [])
        prerequisites = fm.pop("prerequisites", [])
        skill_type = fm.pop("skill_type", "declarative")

        if not isinstance(tools, list):
            tools = [tools]
        if not isinstance(env, list):
            env = [env]
        if not isinstance(prerequisites, list):
            prerequisites = [prerequisites]

        # Check if skill with same name already exists
        existing = await skill_service.get_skill_by_name(name=name, tenant_id=tenant_id)
        if existing:
            raise ApiException(code=409, message=f"Skill '{name}' already exists. Please delete it first or use a different name.")

        skill_data = SkillCreate(
            name=name,
            description=description,
            skill_type=skill_type,
            content=body,
            metadata_json=json.dumps(fm, ensure_ascii=False) if fm else None,
            required_tools=json.dumps(tools, ensure_ascii=False) if tools else None,
            required_env=json.dumps(env, ensure_ascii=False) if env else None,
            prerequisites=json.dumps(
                [p if isinstance(p, dict) else {"description": str(p)} for p in prerequisites],
                ensure_ascii=False,
            ) if prerequisites else None,
            enabled=True,
        )

        skill = await skill_service.create_skill(skill_data=skill_data, tenant_id=tenant_id)
        return success_response(data=skill, message=f"Skill '{name}' installed successfully.")

    except ApiException:
        raise
    except UnicodeDecodeError:
        raise ApiException(code=400, message="File encoding error. Please ensure the file is UTF-8 encoded.")
    except Exception as e:
        logger.error(f"Failed to install skill: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to install skill: {e}")


@skill_router.post("/install-folder")
async def install_skill_folder(
    files: list[UploadFile] = File(...),
    folder_name: Optional[str] = Form(None),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    skill_service: SkillService = Depends(get_skill_service),
):
    """Install a skill from an uploaded folder (multiple files).

    Accepts multiple files that represent a skill directory.
    At least one .md file must be present.
    """
    if not files:
        raise ApiException(code=400, message="At least one file is required.")

    # Check for at least one .md file
    md_files = [f for f in files if f.filename and f.filename.lower().endswith(".md")]
    if not md_files:
        raise ApiException(code=400, message="At least one .md file is required in the skill folder.")

    try:
        # Create a temporary directory to save uploaded files
        with tempfile.TemporaryDirectory() as tmp_dir:
            skill_dir = os.path.join(tmp_dir, folder_name or "skill")
            os.makedirs(skill_dir, exist_ok=True)

            for f in files:
                if f.filename:
                    # Preserve relative path structure
                    file_path = os.path.join(skill_dir, os.path.basename(f.filename))
                    content_bytes = await f.read()
                    with open(file_path, "wb") as fh:
                        fh.write(content_bytes)

            # Use the loader to parse the directory
            parsed = load_skill_from_directory(skill_dir)

            # Check for existing skill
            existing = await skill_service.get_skill_by_name(name=parsed.metadata.name, tenant_id=tenant_id)
            if existing:
                raise ApiException(
                    code=409,
                    message=f"Skill '{parsed.metadata.name}' already exists."
                )

            # Convert to DB model
            db_fields = parsed_skill_to_db_fields(parsed)
            skill_data = SkillCreate(**db_fields)
            skill = await skill_service.create_skill(skill_data=skill_data, tenant_id=tenant_id)

            return success_response(
                data=skill,
                message=f"Skill '{parsed.metadata.name}' installed successfully ({len(files)} files)."
            )

    except ApiException:
        raise
    except Exception as e:
        logger.error(f"Failed to install skill folder: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to install skill folder: {e}")


@skill_router.put("/{skill_id}", response_model=SkillRead)
async def update_skill(
    skill_id: str,
    update_data: SkillUpdate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    skill_service: SkillService = Depends(get_skill_service),
):
    """Update an existing skill."""
    try:
        skill = await skill_service.update_skill(
            skill_id=skill_id, update_data=update_data, tenant_id=tenant_id
        )
        return success_response(data=skill, message="Skill updated successfully.")
    except ValueError as e:
        raise ApiException(code=404, message=str(e))
    except Exception as e:
        logger.error(f"Failed to update skill: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to update skill: {e}")


@skill_router.put("/{skill_id}/toggle", response_model=SkillRead)
async def toggle_skill(
    skill_id: str,
    enabled: bool = Query(...),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    skill_service: SkillService = Depends(get_skill_service),
):
    """Enable or disable a skill."""
    try:
        skill = await skill_service.toggle_skill(
            skill_id=skill_id, enabled=enabled, tenant_id=tenant_id
        )
        status = "enabled" if enabled else "disabled"
        return success_response(data=skill, message=f"Skill {status} successfully.")
    except ValueError as e:
        raise ApiException(code=404, message=str(e))
    except Exception as e:
        logger.error(f"Failed to toggle skill: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to toggle skill: {e}")


@skill_router.delete("/{skill_id}")
async def delete_skill(
    skill_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    skill_service: SkillService = Depends(get_skill_service),
):
    """Delete a skill."""
    try:
        await skill_service.delete_skill(skill_id=skill_id, tenant_id=tenant_id)
        return success_response(message="Skill deleted successfully.")
    except ValueError as e:
        raise ApiException(code=404, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete skill: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to delete skill: {e}")
