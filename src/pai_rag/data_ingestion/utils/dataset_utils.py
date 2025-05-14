import asyncio
import os
from typing import List
import pathlib
from pai_rag.integrations.readers.pai.constants import ACCEPTABLE_DOC_TYPES
from loguru import logger

from pai_rag.integrations.vector_stores.elasticsearch.my_elasticsearch import (
    MyElasticsearchStore,
)


def get_input_files(
    file_path_or_directory: str | List[str],
    filter_pattern: str = None,
    supported_file_types: List[str] = ACCEPTABLE_DOC_TYPES,
):
    logger.info(f"Getting input files from {file_path_or_directory}.")
    filter_pattern = filter_pattern or "*"

    input_files = None
    if isinstance(file_path_or_directory, list):
        # file list
        input_files = [
            str(f)
            for f in file_path_or_directory
            if os.path.isfile(f)
            and pathlib.Path(f).suffix.lower() in supported_file_types
        ]
    elif isinstance(file_path_or_directory, str) and os.path.isdir(
        file_path_or_directory
    ):
        # glob from directory
        directory = pathlib.Path(file_path_or_directory)
        input_files = [
            str(f)
            for f in directory.rglob(filter_pattern)
            if os.path.isfile(f)
            and pathlib.Path(f).suffix.lower() in supported_file_types
        ]
    elif pathlib.Path(file_path_or_directory).suffix.lower() in supported_file_types:
        # Single file
        input_files = [str(file_path_or_directory)]
    else:
        if not os.path.exists(file_path_or_directory):
            raise ValueError(f"Input path '{file_path_or_directory}' does not exist.")
        raise ValueError(
            f"Invalid input path or not supported file type for '{file_path_or_directory}'."
        )

    if not input_files:
        logger.warning(
            f"No file found at path '{file_path_or_directory}' with pattern '{filter_pattern}'."
        )
        return []

    logger.info(
        f"Found {len(input_files)} files at path '{file_path_or_directory}' with pattern '{filter_pattern}'. Samples: {input_files[:5]}"
    )
    return input_files


async def check_single_file(
    es_store: MyElasticsearchStore,
    file_name: str,
    semaphore: asyncio.Semaphore,
):
    file_name = os.path.basename(file_name)
    async with semaphore:
        result = await es_store.client.search(
            index=es_store.index_name,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "term": {
                                    "metadata.file_name.keyword": {"value": file_name}
                                }
                            }
                        ]
                    }
                }
            },
        )
        chunk_count = len(result["hits"]["hits"])
        logger.info(
            f"Searched {file_name} in ES index {es_store.index_name}, chunk count: {chunk_count}"
        )
        return chunk_count == 0


async def filter_files_in_es(
    es_store: MyElasticsearchStore, candidate_file_list: List[str] = []
):
    if not candidate_file_list:
        return candidate_file_list

    semaphore = asyncio.Semaphore(30)
    tasks = [
        check_single_file(es_store, file_name, semaphore)
        for file_name in candidate_file_list
    ]
    results = await asyncio.gather(*tasks)

    real_files = []
    for i, not_exist in enumerate(results):
        if not_exist:
            real_files.append(candidate_file_list[i])
        else:
            logger.info(
                f"File {candidate_file_list[i]} already exists in ES, skipping."
            )

    return real_files


def get_input_files_with_es_backend(
    es_store: MyElasticsearchStore,
    file_path_or_directory: str | List[str],
    filter_pattern: str = None,
):
    candidate_file_list = get_input_files(
        file_path_or_directory=file_path_or_directory, filter_pattern=filter_pattern
    )

    return asyncio.run(
        filter_files_in_es(es_store=es_store, candidate_file_list=candidate_file_list)
    )
