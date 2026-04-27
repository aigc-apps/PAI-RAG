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


def ensure_memory_scope(scope):
    os.makedirs(scope.root, exist_ok=True)
    os.makedirs(scope.archive_dir, exist_ok=True)
    if not os.path.exists(scope.index_path):
        with open(scope.index_path, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_INDEX)
    if not os.path.exists(scope.facts_path):
        with open(scope.facts_path, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_FACTS)


def read_index(scope):
    ensure_memory_scope(scope)
    with open(scope.index_path, encoding='utf-8') as f:
        return f.read()
