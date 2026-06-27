from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import List
from loguru import logger
from agent.tools.registry import ToolRegistry


def load_skills(path: str, registry: ToolRegistry) -> List[str]:
    """Import each *.py skill under `path` exposing `get_tools() -> list[Tool]` and
    register its tools. Returns registered tool names. A skill that fails to import
    is logged and skipped (one bad skill never breaks boot). Files starting with
    '_' are ignored."""
    registered: List[str] = []
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return registered
    for py in sorted(root.glob("*.py")):
        if py.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"skill_{py.stem}", py)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            get_tools = getattr(mod, "get_tools", None)
            if get_tools is None:
                logger.warning(f"skill {py.name} has no get_tools(); skipping")
                continue
            for tool in get_tools():
                registry.register(tool)
                registered.append(tool.name)
        except Exception:
            logger.exception(f"failed to load skill {py}")
    return registered
