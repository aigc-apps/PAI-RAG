"""Stateless text extraction for the new File resource.

Extraction returns as much text as the extractor can produce, bounded by a
safety cap so a pathologically large file can't blow up a DB row. Slicing
for clients happens at read time (`GET /v1/files/{id}/text?offset&limit`),
not here — so the stored blob is the single source of truth.
"""
from io import BytesIO
from typing import BinaryIO, Optional, Tuple

import pandas as pd


EXTRACTOR_VERSION = "v2"

# Per-extraction hard cap. TEXT columns in MySQL hold ~64KB but most deployments
# use utf8mb4 with a larger column type; 500KB is comfortable for SQLite and
# postgres, and can be raised with a column-type migration if needed.
MAX_EXTRACT_CHARS = 500_000

_TEXT_EXTENSIONS = {
    ".txt", ".md", ".json", ".jsonl", ".yaml", ".yml",
    ".xml", ".log", ".py", ".js", ".ts", ".html", ".css",
}


def _cap(text: str) -> Tuple[str, bool]:
    if len(text) > MAX_EXTRACT_CHARS:
        return text[:MAX_EXTRACT_CHARS], True
    return text, False


def extract_text(
    file_data: BinaryIO,
    file_extension: str,
) -> Optional[Tuple[str, bool]]:
    """Extract text from a file. Returns (content, truncated_at_extract).

    `truncated_at_extract=True` means the source exceeded MAX_EXTRACT_CHARS
    and only the head is stored. Clients paginating via ?offset=&limit= will
    still hit the wall at MAX_EXTRACT_CHARS. Re-extraction with a larger cap
    is the escape hatch (not in scope for M1).

    Returns None for formats with no cheap inline preview (binary/multimodal).
    """
    ext = (file_extension or "").lower()

    if ext in (".xlsx", ".xls"):
        file_data.seek(0)
        df = pd.read_excel(file_data)
        # Serialize the whole sheet; the cap trims anything absurd.
        return _cap(df.to_csv(index=False))

    if ext == ".csv":
        file_data.seek(0)
        df = pd.read_csv(file_data)
        return _cap(df.to_csv(index=False))

    if ext in _TEXT_EXTENSIONS:
        file_data.seek(0)
        raw = file_data.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
        return _cap(text)

    return None


def extract_text_from_bytes(
    raw: bytes, file_extension: str
) -> Optional[Tuple[str, bool]]:
    return extract_text(BytesIO(raw), file_extension)
