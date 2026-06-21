import re
import time

from db.models.knowledgebase.file import KbFileEntity
from pairag.file.models.file_item import FileItem

# first markdown ATX heading
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.M)
_TEXT_TITLE_EXTS = (".md", ".markdown")


def _derive_title(file_item: FileItem) -> str:
    """A display title for a manually-uploaded file.

    Defaults to the original filename; for markdown, prefer the first heading.
    Best-effort — never raises (falls back to the filename).
    """
    name = file_item.file_name or ""
    ext = (file_item.file_extension or "").lower()
    if ext in _TEXT_TITLE_EXTS:
        try:
            f = file_item.file
            f.seek(0)
            data = f.read()
            f.seek(0)
            text = data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)
            m = _HEADING_RE.search(text)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
    return name


def to_file_entity(file_item: FileItem) -> KbFileEntity:
        metadata = file_item.metadata()
        # Ensure every file has a human title so display/search is uniform with
        # data-source files (which always carry one). Don't clobber a caller value.
        metadata.setdefault("title", _derive_title(file_item))
        return KbFileEntity(
            id=file_item.id,
            kb_id=file_item.kb_id,
            file_name=file_item.file_name,
            file_size=file_item.file_size,
            file_extension=file_item.file_extension,
            file_path=file_item.file_path,
            file_md5=file_item.file_md5,
            file_metadata=metadata,
            message_id=f"tmp-{int(time.time())}",
            file_content="",
            file_content_length=0,
            tenant_id=file_item.tenant_id,
        )
