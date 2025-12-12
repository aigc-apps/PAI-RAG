from db.models.knowledgebase.file import KbFileEntity
from pairag.file.models.file_item import FileItem
import time

def to_file_entity(file_item: FileItem) -> KbFileEntity:
        return KbFileEntity(
            id=file_item.id,
            kb_id=file_item.kb_id,
            file_name=file_item.file_name,
            file_size=file_item.file_size,
            file_extension=file_item.file_extension,
            file_path=file_item.file_path,
            file_md5=file_item.file_md5,
            file_metadata=file_item.metadata(),
            message_id=f"tmp-{int(time.time())}",
            file_content="",
            file_content_length=0,
            tenant_id=file_item.tenant_id,
        )
