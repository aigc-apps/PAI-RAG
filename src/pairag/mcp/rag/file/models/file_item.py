import hashlib
from typing import Any, BinaryIO, Dict, List
import os
import uuid

from pairag.db.models.knowledgebase.file import KbFileEntity
from pairag.mcp.rag.file.file_utils import ensure_file_type_is_supported


class FileItem:
    def __init__(
        self,
        id: str,
        file_path: str,
        file: BinaryIO,
        kb_id: str,
        file_extension: str,
        file_name: str,
        file_md5: str,
        file_size: int,
    ):
        self.id = id
        self.file_path = file_path
        self.kb_id = kb_id
        self.file_extension = file_extension
        self.file_name = file_name
        self.file_md5 = file_md5
        self.file_size = file_size
        self.file = file

    @classmethod
    def from_file(
        cls,
        file_path,
        file: BinaryIO,
        kb_id: str,
    ):
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1]
        ensure_file_type_is_supported(file_extension)

        file.seek(0)
        file_data = file.read()
        file_md5 = hashlib.md5(file_data).hexdigest()
        id = uuid.uuid4().hex
        file_size = len(file_data)
        return cls(
            id=id,
            file_extension=file_extension,
            file_name=file_name,
            file_md5=file_md5,
            file_path=file_path,
            file_size=file_size,
            kb_id=kb_id,
            file=file,
        )

    @classmethod
    def from_path(
        cls,
        file_path,
        kb_id: str,
    ):
        with open(file_path, "rb") as file:
            return FileItem.from_file(
                file_path=file_path,
                file=file,
                kb_id=kb_id,
            )

    def metadata(self) -> Dict[str, Any]:
        # TODO: maybe create time / modified time?
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_extension": self.file_extension,
            "doc_id": self.id,
        }

    def get_data(self) -> List[bytes]:
        self.file.seek(0)
        return self.file.read()

    def to_file_entity(self) -> KbFileEntity:
        return KbFileEntity(
            id=self.id,
            kb_id=self.kb_id,
            file_name=self.file_name,
            file_size=self.file_size,
            file_extension=self.file_extension,
            file_path=self.file_path,
            file_md5=self.file_md5,
            file_metadata=self.metadata(),
        )
