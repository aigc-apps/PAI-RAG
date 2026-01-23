import hashlib
from typing import Any, BinaryIO, Dict, List, Optional
import os
import io
import uuid
from pairag.file.utils.file_utils import ensure_file_type_is_supported


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
        tenant_id: Optional[str] = None,
    ):
        self.id = id
        self.file_path = file_path
        self.kb_id = kb_id
        self.file_extension = file_extension.lower()
        self.file_name = file_name
        self.file_md5 = file_md5
        self.file_size = file_size
        self.file = file
        self.tenant_id = tenant_id

    @classmethod
    def from_file(
        cls,
        file_path,
        file: BinaryIO,
        kb_id: str,
        file_name: str = None,
        tenant_id: Optional[str] = None,
    ):
        if not file_name:
            file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
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
            tenant_id=tenant_id,
        )

    @classmethod
    def from_path(
        cls,
        file_path,
        kb_id: str,
        tenant_id: Optional[str] = None,
    ):
        with open(file_path, "rb") as f:
            file_content = f.read() 

        file = io.BytesIO(file_content)

        return FileItem.from_file(
            file_path=file_path,
            file=file,
            kb_id=kb_id,
            tenant_id=tenant_id,
        )

    def metadata(self) -> Dict[str, Any]:
        # TODO: maybe create time / modified time?
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_extension": self.file_extension,
        }

    def get_data(self) -> List[bytes]:
        self.file.seek(0)
        return self.file.read()
