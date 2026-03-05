"""Skill Loader for PAI-RAG.

Parses .md skill files with YAML frontmatter and converts them into:
1. Prompt instruction blocks (injected into system prompt)
2. Metadata for prerequisite validation
3. Tool requirement declarations

Skill .md format:
---
name: my-skill
description: What this skill does
tools:
  - http_request
  - run_command
env:
  - API_KEY
  - SECRET_TOKEN
prerequisites:
  - description: "MCP X must be connected"
    check_type: mcp_connection
    params:
      mcp_name: "some_mcp"
  - description: "Python >= 3.10"
    check_type: command
    params:
      command: "python3 --version"
---

# Skill Title

Actual instructions for the LLM on how to use this skill...
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

import yaml
from loguru import logger


@dataclass
class SkillMetadata:
    """Parsed skill metadata from YAML frontmatter."""
    name: str
    description: str = ""
    tools: List[str] = field(default_factory=list)
    env: List[str] = field(default_factory=list)
    prerequisites: List[Dict[str, Any]] = field(default_factory=list)
    skill_type: str = "declarative"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedSkill:
    """A fully parsed skill, ready for use."""
    metadata: SkillMetadata
    content: str  # The markdown body (instructions for LLM)
    source_path: str  # File or directory path
    extra_files: List[str] = field(default_factory=list)  # Non-.md files in a skill package


# --- Frontmatter Parsing ---

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$",
    re.DOTALL,
)


def parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from a markdown string.

    Returns:
        (frontmatter_dict, body_text)
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        # No frontmatter found — treat entire text as body
        return {}, text.strip()

    yaml_str = match.group(1)
    body = match.group(2).strip()

    try:
        fm = yaml.safe_load(yaml_str)
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML frontmatter: {e}")
        fm = {}

    return fm, body


def _extract_metadata(fm: Dict[str, Any]) -> SkillMetadata:
    """Extract SkillMetadata from frontmatter dict."""
    name = fm.pop("name", "unnamed-skill")
    description = fm.pop("description", "")
    tools = fm.pop("tools", [])
    env = fm.pop("env", [])
    prerequisites = fm.pop("prerequisites", [])

    # Determine skill type from frontmatter or auto-detect
    skill_type = fm.pop("skill_type", "declarative")

    if not isinstance(tools, list):
        tools = [tools]
    if not isinstance(env, list):
        env = [env]
    if not isinstance(prerequisites, list):
        prerequisites = [prerequisites]

    return SkillMetadata(
        name=name,
        description=description,
        tools=tools,
        env=env,
        prerequisites=prerequisites,
        skill_type=skill_type,
        extra=fm,  # remaining keys
    )


# --- File-based Loading ---

def load_skill_from_file(file_path: str) -> ParsedSkill:
    """Load a single .md skill file.

    Args:
        file_path: Path to the .md file.

    Returns:
        ParsedSkill instance.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {file_path}")
    if not path.suffix.lower() == ".md":
        raise ValueError(f"Skill file must be .md: {file_path}")

    text = path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    metadata = _extract_metadata(fm)

    # Use filename (without ext) as fallback name
    if metadata.name == "unnamed-skill":
        metadata.name = path.stem

    return ParsedSkill(
        metadata=metadata,
        content=body,
        source_path=str(path),
    )


def load_skill_from_directory(dir_path: str) -> ParsedSkill:
    """Load a skill from a directory (composite skill package).

    A composite skill directory must contain at least one .md file.
    Other files (Python scripts, configs, etc.) are tracked as extra_files.

    Args:
        dir_path: Path to the skill directory.

    Returns:
        ParsedSkill instance.
    """
    path = Path(dir_path)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    # Find .md files
    md_files = sorted(path.glob("*.md"))
    if not md_files:
        raise ValueError(f"No .md file found in skill directory: {dir_path}")

    # Use the first .md file as the primary skill definition
    # Prefer files named SKILL.md, README.md, or index.md
    primary_md = md_files[0]
    for preferred in ["SKILL.md", "skill.md", "README.md", "index.md"]:
        candidate = path / preferred
        if candidate.exists():
            primary_md = candidate
            break

    text = primary_md.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    metadata = _extract_metadata(fm)

    if metadata.name == "unnamed-skill":
        metadata.name = path.name

    # Collect extra files (non-.md)
    extra_files = []
    for f in path.rglob("*"):
        if f.is_file() and f.suffix.lower() != ".md":
            extra_files.append(str(f))

    # If extra files exist, mark as composite
    if extra_files and metadata.skill_type == "declarative":
        metadata.skill_type = "composite"

    # If prerequisites mention commands, mark as command type
    for prereq in metadata.prerequisites:
        if isinstance(prereq, dict) and prereq.get("check_type") == "command":
            metadata.skill_type = "command"
            break

    return ParsedSkill(
        metadata=metadata,
        content=body,
        source_path=str(path),
        extra_files=extra_files,
    )


def scan_skills_directory(skills_dir: str) -> List[ParsedSkill]:
    """Scan a directory for skills.

    Looks for:
    - .md files directly in the directory (standalone skills)
    - Subdirectories containing .md files (composite skill packages)

    Args:
        skills_dir: Root skills directory path.

    Returns:
        List of ParsedSkill instances.
    """
    path = Path(skills_dir)
    if not path.exists():
        logger.warning(f"Skills directory does not exist: {skills_dir}")
        return []

    skills = []

    # Standalone .md files
    for md_file in sorted(path.glob("*.md")):
        try:
            skill = load_skill_from_file(str(md_file))
            skills.append(skill)
            logger.info(f"Loaded standalone skill: {skill.metadata.name} from {md_file}")
        except Exception as e:
            logger.warning(f"Failed to load skill from {md_file}: {e}")

    # Subdirectories (composite skills)
    for sub_dir in sorted(path.iterdir()):
        if sub_dir.is_dir() and not sub_dir.name.startswith("."):
            try:
                skill = load_skill_from_directory(str(sub_dir))
                skills.append(skill)
                logger.info(f"Loaded composite skill: {skill.metadata.name} from {sub_dir}")
            except Exception as e:
                logger.warning(f"Failed to load skill from {sub_dir}: {e}")

    return skills


# --- Prompt Generation ---

def skill_to_prompt_block(skill: ParsedSkill) -> str:
    """Convert a ParsedSkill to a prompt instruction block.

    This is the text that gets injected into the system prompt to teach
    the LLM how to use this skill.

    Args:
        skill: The parsed skill.

    Returns:
        Formatted prompt block string.
    """
    lines = []
    lines.append(f"### Skill: {skill.metadata.name}")
    if skill.metadata.description:
        lines.append(f"**Description**: {skill.metadata.description}")

    if skill.metadata.tools:
        lines.append(f"**Required Tools**: {', '.join(skill.metadata.tools)}")

    if skill.metadata.prerequisites:
        prereq_texts = []
        for prereq in skill.metadata.prerequisites:
            if isinstance(prereq, dict):
                prereq_texts.append(prereq.get("description", str(prereq)))
            else:
                prereq_texts.append(str(prereq))
        lines.append(f"**Prerequisites**: {'; '.join(prereq_texts)}")

    lines.append("")
    lines.append(skill.content)
    lines.append("")

    return "\n".join(lines)


def build_skills_prompt_section(skills: List[ParsedSkill]) -> str:
    """Build the full skills section for system prompt injection.

    Args:
        skills: List of enabled ParsedSkill instances.

    Returns:
        Complete skills prompt section, or empty string if no skills.
    """
    if not skills:
        return ""

    blocks = []
    blocks.append("## 🛠️ Installed Skills")
    blocks.append("")
    blocks.append("You have the following skills installed. When a user's request matches a skill's purpose, follow the skill's instructions to complete the task using the available tools.")
    blocks.append("")

    for skill in skills:
        blocks.append(skill_to_prompt_block(skill))

    return "\n".join(blocks)


# --- Prerequisite Validation ---

def validate_prerequisites(
    skill: ParsedSkill,
    available_tools: List[str] = None,
    env_vars: Dict[str, str] = None,
) -> List[str]:
    """Validate a skill's prerequisites.

    Args:
        skill: The skill to validate.
        available_tools: List of currently available tool names.
        env_vars: Current environment variables (defaults to os.environ).

    Returns:
        List of validation error messages. Empty list means all prerequisites met.
    """
    errors = []

    if available_tools is None:
        available_tools = []
    if env_vars is None:
        env_vars = dict(os.environ)

    # Check required tools
    for tool_name in skill.metadata.tools:
        if tool_name not in available_tools:
            errors.append(f"Required tool '{tool_name}' is not available.")

    # Check required env vars
    for env_key in skill.metadata.env:
        if env_key not in env_vars or not env_vars[env_key]:
            errors.append(f"Required environment variable '{env_key}' is not set.")

    # Check explicit prerequisites
    for prereq in skill.metadata.prerequisites:
        if isinstance(prereq, dict):
            check_type = prereq.get("check_type", "")
            description = prereq.get("description", "Unknown prerequisite")

            if check_type == "env":
                var_name = prereq.get("params", {}).get("var_name", "")
                if var_name and (var_name not in env_vars or not env_vars[var_name]):
                    errors.append(f"Prerequisite not met: {description}")

            elif check_type == "mcp_connection":
                # MCP connection checks are deferred to runtime
                # Just note them as informational
                logger.info(f"Skill '{skill.metadata.name}' requires MCP: {description}")

            elif check_type == "command":
                # Command prerequisite checks are deferred to runtime
                logger.info(f"Skill '{skill.metadata.name}' requires command check: {description}")

    return errors


# --- Conversion Utilities ---

def parsed_skill_to_db_fields(skill: ParsedSkill) -> Dict[str, Any]:
    """Convert a ParsedSkill to a dict suitable for SkillCreate model.

    Args:
        skill: Parsed skill instance.

    Returns:
        Dict matching SkillCreate fields.
    """
    return {
        "name": skill.metadata.name,
        "description": skill.metadata.description,
        "skill_type": skill.metadata.skill_type,
        "content": skill.content,
        "metadata_json": json.dumps(skill.metadata.extra, ensure_ascii=False) if skill.metadata.extra else None,
        "required_tools": json.dumps(skill.metadata.tools, ensure_ascii=False) if skill.metadata.tools else None,
        "required_env": json.dumps(skill.metadata.env, ensure_ascii=False) if skill.metadata.env else None,
        "prerequisites": json.dumps(
            [p if isinstance(p, dict) else {"description": str(p)} for p in skill.metadata.prerequisites],
            ensure_ascii=False,
        ) if skill.metadata.prerequisites else None,
        "folder_path": skill.source_path if skill.metadata.skill_type != "declarative" else None,
        "enabled": True,
    }
