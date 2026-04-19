import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional
from fastapi import UploadFile
from pairag.file.models.file_item import FileItem
from pairag.file.store.file_store_helper import file_store
from pydantic import BaseModel
from loguru import logger
import json


@dataclass
class StoredFileInfo:
    """Result of uploading an UploadFile to the file_store.

    Decoupled from KbFileEntity/FileItem so the new /v1/files resource can reuse it
    without dragging in knowledgebase semantics (kb_id).
    """
    file_name: str
    file_path: str        # stored path inside file_store
    file_extension: str   # lowercased, includes leading dot
    file_md5: str
    file_size: int


@dataclass
class UploadPreview:
    """md5/size/extension peeked from an UploadFile without hitting storage.
    Useful for dedup lookups before paying the OSS write cost.
    """
    file_name: str
    file_extension: str
    file_md5: str
    file_size: int


def preview_upload(upload: UploadFile) -> UploadPreview:
    file_name = upload.filename
    file_data = upload.file
    file_data.seek(0)
    raw = file_data.read()
    file_md5 = hashlib.md5(raw).hexdigest()
    file_size = len(raw)
    file_data.seek(0)
    extension = os.path.splitext(file_name)[1].lower()
    return UploadPreview(
        file_name=file_name,
        file_extension=extension,
        file_md5=file_md5,
        file_size=file_size,
    )


async def write_upload_to_store(
    upload: UploadFile,
    destination_path: str,
    tenant_id: str,
) -> StoredFileInfo:
    """Low-level helper: persist an UploadFile to the file_store and return
    only the facts we care about (md5/size/ext). Used by both the legacy
    `upload_form_files_async` path and the new `FileResourceService`.
    """
    preview = preview_upload(upload)
    upload_result = await file_store.write_async(
        file=upload.file,
        file_name=preview.file_name,
        file_path=destination_path,
        tenant_id=tenant_id,
    )
    return StoredFileInfo(
        file_name=preview.file_name,
        file_path=upload_result.file_path,
        file_extension=preview.file_extension,
        file_md5=preview.file_md5,
        file_size=preview.file_size,
    )


async def upload_form_files_async(
    kb_id: str,
    files: List[UploadFile],
    tenant_id: str,
) -> List[FileItem]:
    file_items = []
    for single_file in files:
        logger.info(f"Uploading file {single_file.filename} to tenant_id {tenant_id}...")
        file_name = single_file.filename
        destination_file_path = f"{kb_id}/docs/{file_name}"

        stored = await write_upload_to_store(
            upload=single_file,
            destination_path=destination_file_path,
            tenant_id=tenant_id,
        )
        file_item = FileItem.from_file(
            file=single_file.file,
            file_path=stored.file_path,
            kb_id=kb_id,
            file_name=file_name,
            tenant_id=tenant_id,
        )
        file_items.append(file_item)

        logger.info(f"Uploaded file {file_name} to {destination_file_path} to tenant_id {tenant_id} successfully.")

    return file_items


class ParseFileTask(BaseModel):
    file_name: str
    file_path: str


class StartParseTaskRequest(BaseModel):
    files: List[ParseFileTask]
    chunk_config: Optional[dict] = None


async def upload_file_names_async(
    kb_id: str,
    parse_tasks: List[ParseFileTask],
    tenant_id: str,
) -> List[FileItem]:
    file_items = []
    for file_task in parse_tasks:
        file_path = file_task.file_path
        file_name = file_task.file_name
        logger.info(f"Retrieving file {file_name} from file_store to tenant_id {tenant_id}...")
        file = await file_store.read_async(file_path=file_path, tenant_id=tenant_id)
        file_item = FileItem.from_file(file_path=file_path, file=file, kb_id=kb_id, file_name=file_name, tenant_id=tenant_id)
        file_items.append(file_item)
        logger.info(f"Retrieved file {file_name} from file_store to tenant_id {tenant_id} successfully.")
    return file_items


def load_eval_dataset_from_local_path(file_path: str) -> List[dict]:
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    entry_data = json.loads(line)
                    if "input" in entry_data:  # 只有包含 "input" 的才保留
                        results.append(entry_data)
                    else:
                        logger.warning(f"Warning: Line {line_num} missing 'input' field, skipped.")
                except json.JSONDecodeError as e:
                    logger.warning(f"Warning: Line {line_num} is not valid JSON, skipped. Error: {e}")
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found.")
        raise
    except Exception as e:
        logger.error(f"Fail to read file '{file_path}': {e}")
        raise

    return results
