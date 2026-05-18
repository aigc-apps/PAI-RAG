"""Markdown archive writer for completed/updated chat sessions.

SQLite remains the source of truth for active sessions. This module writes a
human-readable copy under ``memory/L4_raw_sessions`` so operators can replay
new sessions without opening the database.
"""
import hashlib
import json
import os
import re
from datetime import datetime


def archive_session_record(record, archive_dir, *, created_after=None):
    """Write ``record`` to ``archive_dir`` and return the archive path.

    Empty sessions are intentionally ignored; this is a forward-only archive
    for newly updated conversations, not a backfill mechanism.
    """
    if not record:
        return ''
    messages = record.get('ui_messages') or []
    if not messages:
        return ''
    os.makedirs(archive_dir, exist_ok=True)
    path = os.path.join(
        archive_dir,
        archive_filename(record.get('session_id') or '', record.get('created_at') or record.get('updated_at') or ''),
    )
    if created_after is not None and not os.path.exists(path):
        created_at = _parse_datetime(record.get('created_at') or record.get('updated_at') or '')
        boundary = _coerce_datetime(created_after)
        if created_at is None or boundary is None or created_at < boundary:
            return ''
    _write_text_atomic(path, render_session_archive(record))
    return path


def archive_filename(session_id, created_at):
    stamp = _compact_timestamp(created_at)
    suffix = _safe_suffix(session_id)
    return f'{stamp}_{suffix}.md'


def render_session_archive(record):
    messages = record.get('ui_messages') or []
    created_at = record.get('created_at') or ''
    updated_at = record.get('updated_at') or ''
    stamp = _compact_timestamp(created_at or updated_at)
    first_user = _first_user_text(messages)
    metadata = {
        'session_id': record.get('session_id') or '',
        'user_id': record.get('user_id') or '',
        'title': record.get('title') or '',
        'status': record.get('status') or '',
        'created_at': created_at,
        'updated_at': updated_at,
        'message_count': len(messages),
        'workspace_path': record.get('workspace_path') or '',
    }
    history = [
        {
            'role': msg.get('role') or '',
            'content': _message_text(msg.get('content')),
        }
        for msg in messages
        if isinstance(msg, dict)
    ]
    return '\n'.join([
        f'# Task ({stamp})',
        first_user or '(empty)',
        '',
        '## Metadata',
        '```json',
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        '```',
        '',
        '## Transcript',
        _render_transcript(messages),
        '',
        '## History',
        '```json',
        json.dumps(history, ensure_ascii=False, indent=2, default=str),
        '```',
        '',
        '## UI Messages',
        '```json',
        json.dumps(messages, ensure_ascii=False, indent=2, default=str),
        '```',
        '',
    ])


def _compact_timestamp(value):
    parsed = _parse_datetime(value)
    if parsed is not None:
        return parsed.strftime('%Y%m%d_%H%M%S')
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _coerce_datetime(value):
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value
    return _parse_datetime(value)


def _parse_datetime(value):
    text = str(value or '').strip()
    if not text:
        return None
    if text.endswith('Z'):
        text = f'{text[:-1]}+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.replace(tzinfo=None)
    return parsed


def _safe_suffix(session_id):
    text = re.sub(r'[^A-Za-z0-9_-]+', '_', str(session_id or '').strip()).strip('_')
    if text:
        return text[:8]
    return hashlib.sha1(os.urandom(16)).hexdigest()[:8]


def _first_user_text(messages):
    for msg in messages:
        if isinstance(msg, dict) and msg.get('role') == 'user':
            return _message_text(msg.get('content')).strip()
    return ''


def _render_transcript(messages):
    parts = []
    for index, msg in enumerate(messages or [], start=1):
        if not isinstance(msg, dict):
            continue
        role = (msg.get('role') or 'message').strip().title()
        content = _message_text(msg.get('content')).strip()
        if not content:
            content = '(no visible content)'
        parts.extend([f'### {index}. {role}', content, ''])
    return '\n'.join(parts).rstrip() or '(empty)'


def _message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get('type') in ('text', 'input_text', 'output_text'):
                    parts.append(str(block.get('text') or ''))
                elif 'content' in block:
                    parts.append(str(block.get('content') or ''))
        return '\n'.join(part for part in parts if part)
    if content is None:
        return ''
    return str(content)


def _write_text_atomic(path, content):
    tmp_path = f'{path}.tmp.{os.getpid()}'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(content)
    os.replace(tmp_path, path)
