from db.models.knowledgebase.file import KbFileEntity
from pairag.file.models.file_item import FileItem
import time

def to_file_entity(file: FileItem) -> KbFileEntity:
        return KbFileEntity(
            id=file.id,
            kb_id=file.kb_id,
            file_name=file.file_name,
            file_size=file.file_size,
            file_extension=file.file_extension,
            file_path=file.file_path,
            file_md5=file.file_md5,
            file_metadata=file.metadata(),
            message_id=f"tmp-{int(time.time())}",
            file_content="",
            file_content_length=0,
        )
