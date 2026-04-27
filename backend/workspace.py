"""Workspace isolation helpers for multi-user agent execution."""
import os
import re


SAFE_ID_RE = re.compile(r'[^A-Za-z0-9_.-]+')


class WorkspaceViolation(ValueError):
    """Raised when a requested path escapes the assigned workspace."""


def safe_segment(value):
    value = SAFE_ID_RE.sub('_', str(value or '').strip())
    return value.strip('._') or 'default'


def is_relative_to(path, root):
    path = os.path.abspath(path)
    root = os.path.abspath(root)
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


class WorkspaceManager:
    def __init__(self, root):
        self.root = os.path.abspath(root)

    def session_root(self, user_id, session_id):
        return os.path.join(self.root, safe_segment(user_id), safe_segment(session_id))

    def prepare_session(self, user_id, session_id, cwd=None):
        root = self.session_root(user_id, session_id)
        os.makedirs(root, exist_ok=True)
        if not cwd:
            return root
        candidate = cwd if os.path.isabs(str(cwd)) else os.path.join(root, str(cwd))
        candidate = os.path.abspath(candidate)
        if not is_relative_to(candidate, root):
            raise WorkspaceViolation('workspace_violation: cwd escapes session workspace')
        os.makedirs(candidate, exist_ok=True)
        return candidate
