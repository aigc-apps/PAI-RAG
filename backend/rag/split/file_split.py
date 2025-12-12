# Split form file into multiple parts (csv, excel, jsonl)

import os
from typing import Iterator
import uuid
from common.knowledgebase.types import FileStatus
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.file_task import KbFileTaskEntity
from rag.split.csv_split import split_csv
from rag.split.excel_split import split_excel
from rag.split.jsonl_split import split_jsonl


def split_file_tasks(file_entity: KbFileEntity) -> Iterator[KbFileTaskEntity]:
    _, file_ext = os.path.splitext(file_entity.file_name)

    # 不是表格格式，直接返回
    if file_ext not in [".csv", ".xlsx", ".jsonl", ".xls"]:
        yield KbFileTaskEntity(
            id=uuid.uuid4().hex,
            file_id=file_entity.id,
            kb_id=file_entity.kb_id,
            file_part=0,
            file_path=file_entity.file_path,
            file_version=file_entity.file_version,
            status=FileStatus.pending,
            tenant_id=file_entity.tenant_id,
        )
        return

    if file_ext == ".csv":
        yield from split_csv(file_entity)
    elif file_ext == ".xlsx" or file_ext == ".xls":
        yield from split_excel(file_entity)
    elif file_ext == ".jsonl":
        yield from split_jsonl(file_entity)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}.")
