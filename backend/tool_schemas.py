"""Helpers for loading tool schemas for foreground and background agents."""
import copy
import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAIN_EXCLUDED_TOOLS = {'start_long_term_update'}
BACKGROUND_MEMORY_REVIEW_TOOLS = {
    'start_long_term_update',
    'file_read',
    'file_patch',
    'file_write',
}


def _tool_name(schema):
    return ((schema or {}).get('function') or {}).get('name', '')


def load_tools_schema(names=None, exclude=None):
    path = os.path.join(ROOT, 'tools_schema.json')
    with open(path, encoding='utf-8') as f:
        schemas = json.load(f)
    names = set(names or [])
    exclude = set(exclude or [])
    out = []
    for schema in schemas:
        name = _tool_name(schema)
        if names and name not in names:
            continue
        if exclude and name in exclude:
            continue
        out.append(copy.deepcopy(schema))
    return out


def main_tools_schema():
    return load_tools_schema(exclude=MAIN_EXCLUDED_TOOLS)


def background_memory_review_tools_schema():
    return load_tools_schema(names=BACKGROUND_MEMORY_REVIEW_TOOLS)
