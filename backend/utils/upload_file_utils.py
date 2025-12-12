from typing import List
from fastapi import UploadFile
from pairag.file.models.file_item import FileItem
from pairag.file.store.file_store_helper import file_store
from rag.split.excel_split import convert_xls_to_xlsx
from pydantic import BaseModel
from loguru import logger


async def upload_form_files_async(
    kb_id: str,
    files: List[UploadFile],
    tenant_id: str,
) -> List[FileItem]:
    file_items = []
    for single_file in files:
        logger.info(f"Uploading file {single_file.filename}...")
        file_name = single_file.filename
        file_data = single_file.file
        if single_file.filename.endswith(".xls"):
            file_data = convert_xls_to_xlsx(file_data)
            file_name = file_name[:-4] + ".xlsx"

        destination_file_path = f"{kb_id}/docs/{file_name}"
        upload_result = await file_store.write_async(
            file=file_data,
            file_name=file_name,
            file_path=destination_file_path,
            tenant_id=tenant_id,
        )
        file_item = FileItem.from_file(
            file=file_data,
            file_path=upload_result.file_path,
            kb_id=kb_id,
            file_name=file_name,
            tenant_id=tenant_id,
        )
        file_items.append(file_item)

        logger.info(f"Uploaded file {file_name} to {destination_file_path} successfully.")

    return file_items


class ParseFileTask(BaseModel):
    file_name: str
    file_path: str


class StartParseTaskRequest(BaseModel):
    files: List[ParseFileTask]


async def upload_file_names_async(
    kb_id: str,
    parse_tasks: List[ParseFileTask],
    tenant_id: str,
) -> List[FileItem]:
    file_items = []
    for file_task in parse_tasks:
        file_path = file_task.file_path
        file_name = file_task.file_name
        logger.info(f"Retrieving file {file_name} from file_store...")
        file = await file_store.read_async(file_path=file_path, tenant_id=tenant_id)
        file_item = FileItem.from_file(file_path=file_path, file=file, kb_id=kb_id, file_name=file_name, tenant_id=tenant_id)
        file_items.append(file_item)
        logger.info(f"Retrieved file {file_name} from file_store successfully.")
    return file_items
