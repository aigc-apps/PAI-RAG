"""Read-only inventory for configured and self-evolved skills."""
import os
import re

from skill_manager import scan_skills


EVOLVED_EXTENSIONS = {'.md', '.py'}
EXCLUDED_MEMORY_FILENAMES = {
    'global_index.txt',
    'global_facts.txt',
    'sessions_v2.sqlite3',
}
EXCLUDED_MEMORY_DIRS = {
    'L4_raw_sessions',
    'sessions',
    'users',
    '__pycache__',
}


def _relpath(path, root):
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace(os.sep, '/')


def _first_meaningful_line(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                text = line.strip()
                if not text or text in ('---', '```'):
                    continue
                if re.match(r'^[A-Za-z_-]+:\s*.+$', text):
                    continue
                return re.sub(r'^#+\s*', '', text).strip()
    except Exception:
        return ''
    return ''


def _evolved_kind(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.py':
        return 'script'
    return 'sop'


def official_skills(repo_root):
    skills_root = os.path.join(repo_root, 'skills')
    items = []
    for skill in sorted(scan_skills(skills_root).values(), key=lambda item: item.name):
        items.append({
            'name': skill.name,
            'description': skill.description,
            'trigger': skill.trigger,
            'allowed_tools': list(skill.allowed_tools or []),
            'source': _relpath(skill.path, repo_root),
        })
    return items


def evolved_skills(repo_root, memory_root):
    memory_root = os.path.abspath(memory_root)
    items = []
    if not os.path.isdir(memory_root):
        return items

    for current_root, dirs, files in os.walk(memory_root):
        dirs[:] = sorted(
            name for name in dirs
            if name not in EXCLUDED_MEMORY_DIRS and not name.startswith('.')
        )
        for filename in sorted(files):
            if filename in EXCLUDED_MEMORY_FILENAMES or filename.startswith('.'):
                continue
            path = os.path.join(current_root, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in EVOLVED_EXTENSIONS:
                continue
            source = _relpath(path, repo_root)
            name = os.path.splitext(os.path.basename(path))[0]
            items.append({
                'name': name,
                'description': _first_meaningful_line(path) or name,
                'kind': _evolved_kind(path),
                'source': source,
            })
    return items


def skills_inventory(repo_root, memory_root):
    return {
        'official': official_skills(repo_root),
        'evolved': evolved_skills(repo_root, memory_root),
    }
