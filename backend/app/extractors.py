"""File → Markdown extraction boundary for knowledge-base uploads.

Turns an uploaded file's bytes into Markdown text that the existing ingestion
chain (chunk → embed → rerank) consumes unchanged. Two tiers:

- Plain text / Markdown are decoded directly here (zero dependencies) so uploads
  keep working in the lean base without the `parsers` extra.
- Everything else is handed to Microsoft's `markitdown`, imported lazily inside
  the function so the module (and the whole service) still boots when the
  `parsers` extra is not installed. See pyproject `[project.optional-dependencies]`
  and docs/MIGRATION.md for why heavy parsers stay optional.
"""

from __future__ import annotations

import os
import tempfile

# Decoded directly, no markitdown needed.
TEXT_PASSTHROUGH = {".txt", ".md", ".markdown"}

# Handed to markitdown (needs the `parsers` extra installed).
MARKITDOWN_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".csv",
    ".html",
    ".htm",
    ".json",
    ".xml",
}

SUPPORTED_EXTENSIONS = TEXT_PASSTHROUGH | MARKITDOWN_EXTENSIONS


def _ext(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def extract_to_markdown(filename: str, data: bytes) -> str:
    """Extract an uploaded file's textual content as Markdown.

    Dispatches on the filename extension. Raises ValueError for unsupported
    types, a missing `parsers` extra, or files with no extractable text — the
    caller maps ValueError to HTTP 400. This is synchronous and can block on
    large parses; call it via `asyncio.to_thread` from async code.
    """
    ext = _ext(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported file type '{ext or filename}'")

    if ext in TEXT_PASSTHROUGH:
        return data.decode("utf-8", errors="replace")

    try:
        from markitdown import MarkItDown
    except ImportError as exc:  # extra not installed
        raise ValueError(
            "file parsing requires the 'parsers' extra: uv sync --extra parsers"
        ) from exc

    # markitdown picks a converter by the file's extension, so the temp file
    # must carry the right suffix.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        text = MarkItDown().convert(tmp_path).text_content or ""
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not text.strip():
        raise ValueError("no extractable text in file")
    return text
