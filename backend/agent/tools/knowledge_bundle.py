from __future__ import annotations


KNOWLEDGE_TOOL_NAMES: tuple[str, ...] = (
    "knowledge_search",
    "knowledge_read",
    "knowledge_find",
    "knowledge_list",
)

LEGACY_KNOWLEDGE_TOOL_MAP: dict[str, str] = {
    "view_file": "knowledge_read",
    "grep_file": "knowledge_find",
    "list_knowledge_bases": "knowledge_list",
}


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def normalize_knowledge_tool_lists(
    include: list[str], exclude: list[str]
) -> tuple[list[str], list[str]]:
    knowledge_names = set(KNOWLEDGE_TOOL_NAMES) | set(LEGACY_KNOWLEDGE_TOOL_MAP)
    mapped_include = [
        LEGACY_KNOWLEDGE_TOOL_MAP.get(name, name) for name in include
    ]
    mapped_exclude = [
        LEGACY_KNOWLEDGE_TOOL_MAP.get(name, name) for name in exclude
    ]
    disabled = any(name in knowledge_names for name in exclude)
    nonknowledge_include = [
        name for name in mapped_include if name not in KNOWLEDGE_TOOL_NAMES
    ]
    nonknowledge_exclude = [
        name for name in mapped_exclude if name not in KNOWLEDGE_TOOL_NAMES
    ]
    if disabled:
        return _unique(nonknowledge_include), _unique(
            [*nonknowledge_exclude, *KNOWLEDGE_TOOL_NAMES]
        )
    enabled = any(name in knowledge_names for name in include)
    if enabled:
        return _unique(
            [*nonknowledge_include, *KNOWLEDGE_TOOL_NAMES]
        ), _unique(nonknowledge_exclude)
    return _unique(nonknowledge_include), _unique(nonknowledge_exclude)
