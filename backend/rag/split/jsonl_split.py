# Split csv file into multiple parts

import os
from typing import Iterator, List
import uuid
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.file_task import KbFileTaskEntity
from pairag.file.store.file_store_helper import file_store
from io import TextIOWrapper, BytesIO
from loguru import logger
from rag.split.constants import MAX_PART_ROW_NUM

def _create_file_task(
    file_entity: KbFileEntity,
    current_rows: List[str],
    current_part: int,
    base_path: str,
) -> KbFileTaskEntity:
    file_bytes = "\n".join(current_rows).encode('utf-8')
    binary_buffer = BytesIO(file_bytes)
    file_part_path = f"{base_path}_Part{current_part:04d}.jsonl"
    file_store.save(file=binary_buffer, file_path=file_part_path)
    logger.info(f"Created jsonl part file: {file_part_path} with {len(current_rows)} rows.")
    return KbFileTaskEntity(
        id=uuid.uuid4().hex,
        file_id=file_entity.id,
        kb_id=file_entity.kb_id,
        file_part=current_part,
        file_path=file_part_path,
        file_version=file_entity.file_version,
    )


def split_jsonl(file_entity: KbFileEntity) -> Iterator[KbFileTaskEntity]:
    logger.info(f"Start splitting jsonl file: {file_entity.file_path}")

    file = file_store.load(file_entity.file_path)
    base_path, _ = os.path.splitext(file_entity.file_path)

    text_instream = TextIOWrapper(file, encoding="utf-8")
    current_rows = []
    current_part = 1

    for line in text_instream:
        line = line.strip()
        if not line:
            continue

        if len(current_rows) > 0 and len(current_rows) % MAX_PART_ROW_NUM == 0:
            yield _create_file_task(
                file_entity=file_entity,
                current_rows=current_rows,
                current_part=current_part,
                base_path=base_path,
            )

            current_part += 1
            current_rows = []
        current_rows.append(line)

    # Only one part, rows count less than MAX_PART_ROW_NUM
    if current_part == 1:
        yield KbFileTaskEntity(
            id=uuid.uuid4().hex,
            file_id=file_entity.id,
            kb_id=file_entity.kb_id,
            file_part=0,
            file_path=file_entity.file_path,
            file_version=file_entity.file_version,
        )
    elif current_rows:
        yield _create_file_task(
            file_entity=file_entity,
            current_rows=current_rows,
            current_part=current_part,
            base_path=base_path,
        )
        current_rows = []

    logger.info(f"Finished splitting jsonl file into {current_part} parts.")
    return
