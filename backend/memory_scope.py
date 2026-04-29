"""Long-term memory path isolation for service and user runs."""
import os
from dataclasses import dataclass

from backend.workspace import safe_segment
from session_store import SERVER_USER_ID


DEFAULT_INDEX = "(empty)\n"
DEFAULT_FACTS = "(empty)\n"


@dataclass(frozen=True)
class MemoryScope:
    root: str
    index_path: str
    facts_path: str
    archive_dir: str
    is_global: bool


def memory_scope_for(repo_root, user_id=SERVER_USER_ID):
    memory_root = os.path.join(repo_root, 'memory')
    if (user_id or SERVER_USER_ID) == SERVER_USER_ID:
        root = memory_root
        is_global = True
    else:
        root = os.path.join(memory_root, 'users', safe_segment(user_id))
        is_global = False
    return MemoryScope(
        root=os.path.abspath(root),
        index_path=os.path.abspath(os.path.join(root, 'global_index.txt')),
        facts_path=os.path.abspath(os.path.join(root, 'global_facts.txt')),
        archive_dir=os.path.abspath(os.path.join(root, 'L4_raw_sessions')),
        is_global=is_global,
    )


def _write_if_missing(path, content):
    try:
        with open(path, 'x', encoding='utf-8') as f:
            f.write(content)
    except FileExistsError:
        pass


def _copy_or_write_if_missing(path, source_path, default_content):
    if os.path.exists(path):
        return
    source_exists = source_path and os.path.exists(source_path)
    is_distinct_source = source_path and os.path.abspath(source_path) != os.path.abspath(path)
    if source_exists and is_distinct_source:
        try:
            with open(source_path, 'rb') as src, open(path, 'xb') as dst:
                dst.write(src.read())
            return
        except FileExistsError:
            return
    _write_if_missing(path, default_content)


def _global_memory_root_for(scope):
    if scope.is_global:
        return scope.root
    return os.path.abspath(os.path.join(scope.root, os.pardir, os.pardir))


def ensure_memory_scope(scope):
    os.makedirs(scope.root, exist_ok=True)
    os.makedirs(scope.archive_dir, exist_ok=True)
    global_root = _global_memory_root_for(scope)
    _copy_or_write_if_missing(
        scope.index_path,
        None if scope.is_global else os.path.join(global_root, 'global_index.txt'),
        DEFAULT_INDEX,
    )
    _copy_or_write_if_missing(
        scope.facts_path,
        None if scope.is_global else os.path.join(global_root, 'global_facts.txt'),
        DEFAULT_FACTS,
    )


def read_index(scope):
    ensure_memory_scope(scope)
    with open(scope.index_path, encoding='utf-8') as f:
        return f.read()
