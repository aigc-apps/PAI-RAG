import inspect
import os
from loguru import logger
from ray.data.datasource import ReadTask, Datasource
from ray.air.util.tensor_extensions.arrow import pyarrow_table_from_pydict
from ray.data.block import BlockMetadata
from typing import List, Dict
from pai_rag.data_ingestion.constants import (
    DEFAULT_MODIFIED_AT_FIELD,
    DEFAULT_NODE_SOURCE_FIELD,
)
from pai_rag.data_ingestion.delta.list_delta import (
    list_files_from_langstudio_index,
    list_files_from_rag_service,
)
from pai_rag.data_ingestion.delta.models import DocItem
from pai_rag.data_ingestion.models.config.datasource import DataSourceConfig
from pai_rag.data_ingestion.models.file.event import FileChangeType, NodeOperationType
from pai_rag.data_ingestion.operators.parser import get_path_resolver
from pai_rag.data_ingestion.utils.dataset_utils import get_input_files


"""
计算文件路径和向量数据库中的diff。
"""


class FileDeltaDatasource(Datasource):
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.paths = get_input_files(
            file_path_or_directory=config.input_path,
            supported_file_types=config.file_extensions,
        )

        self.path_resolver = get_path_resolver()
        oss_path_prefix = self.path_resolver.resolve_source_url(config.input_path)

        # default empty
        self.docs_in_store: Dict[str, DocItem] = {}
        if config.enable_delta and config.target_index:
            logger.info(
                f"""Reading existing docs from vector store using langstudio index.
                index_name: {config.target_index}
                version: {config.target_index_version}
            """
            )
            self.docs_in_store = list_files_from_langstudio_index(
                index_name=config.target_index,
                index_version=config.target_index_version,
                oss_path_prefix=oss_path_prefix,
            )
        elif (
            config.enable_delta
            and config.rag_api_key
            and config.rag_endpoint
            and config.knowledgebase
        ):
            logger.info(
                f"""Reading existing docs from vector store using PAI-RAG index.
                rag_endpoint: {config.rag_endpoint}
                knowledgebase: {config.knowledgebase}
                embed_dims: {config.embed_dims}
            """
            )
            self.docs_in_store = list_files_from_rag_service(
                rag_api_key=config.rag_api_key,
                rag_endpoint=config.rag_endpoint,
                knowledgebase=config.knowledgebase,
                embed_dims=config.embed_dims,
                oss_path_prefix=oss_path_prefix,
            )
        elif config.enable_delta:
            logger.error("Please specify the information needed to compute delta.")
            raise ValueError("Please specify the information needed to compute delta.")
        else:
            logger.info("Will not compute delta. This is a full load.")

    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        """
        获取读取任务，返回一个或多个 ReadTask。
        """

        def create_read_task_fn(read_paths):
            def read_files():
                logger.warning(
                    f"read_fn called from:\n{''.join(inspect.stack()[1].code_context or [])}"
                )

                """
                准备读取任务，返回一个或多个 ReadTask。
                """
                # 获取文件夹中的所有文件信息
                print("Starts reading: ", read_paths)
                file_count = 0

                existing_file_set = set()
                for file_path in read_paths:
                    file_name = os.path.basename(file_path)
                    file_uri = self.path_resolver.resolve_source_url(file_path)
                    existing_file_set.add(file_uri)
                    doc_item = self.docs_in_store.get(file_uri)
                    file_mtime = os.stat(file_path).st_mtime

                    if doc_item:
                        print(
                            "matched: ",
                            file_uri,
                            file_name,
                            file_mtime,
                            doc_item.modified_time,
                        )
                        if file_mtime <= doc_item.modified_time:
                            logger.debug(
                                f"Skipping {file_name} as it has not been modified since last indexing."
                            )
                            continue
                        elif file_mtime > doc_item.modified_time:
                            file_count += 1
                            logger.debug(
                                f"{file_name} has been modified since last indexing."
                            )
                            yield pyarrow_table_from_pydict(
                                {
                                    "node_ids": [doc_item.node_ids],
                                    "file_name": [file_name],
                                    "file_path": [file_path],
                                    DEFAULT_NODE_SOURCE_FIELD: [file_uri],
                                    DEFAULT_MODIFIED_AT_FIELD: [file_mtime],
                                    "operation": [NodeOperationType.DELETE],
                                    "operation_reason": [FileChangeType.MODIFY],
                                }
                            )
                            yield pyarrow_table_from_pydict(
                                {
                                    "node_ids": [None],
                                    "file_name": [file_name],
                                    "file_path": [file_path],
                                    DEFAULT_NODE_SOURCE_FIELD: [file_uri],
                                    DEFAULT_MODIFIED_AT_FIELD: [file_mtime],
                                    "operation": [NodeOperationType.ADD],
                                    "operation_reason": [FileChangeType.MODIFY],
                                }
                            )
                    else:
                        logger.debug(f"{file_name} is newly added.")
                        file_count += 1
                        yield pyarrow_table_from_pydict(
                            {
                                "node_ids": [None],
                                "file_name": [file_name],
                                "file_path": [file_path],
                                DEFAULT_NODE_SOURCE_FIELD: [file_uri],
                                DEFAULT_MODIFIED_AT_FIELD: [file_mtime],
                                "operation": [NodeOperationType.ADD],
                                "operation_reason": [FileChangeType.ADD],
                            }
                        )

                for file_uri in self.docs_in_store:
                    if file_uri not in existing_file_set:
                        doc_item = self.docs_in_store[file_uri]
                        logger.info(
                            f"File {file_uri} has been deleted. Will remove from vector store."
                        )
                        yield pyarrow_table_from_pydict(
                            {
                                "node_ids": [doc_item.node_ids],
                                "file_name": [file_uri],
                                "file_path": [file_uri],
                                DEFAULT_NODE_SOURCE_FIELD: [file_uri],
                                DEFAULT_MODIFIED_AT_FIELD: [-1],
                                "operation": [NodeOperationType.DELETE],
                                "operation_reason": [FileChangeType.DELETE],
                            }
                        )
                        file_count += 1

                logger.info(
                    f"Loaded {file_count} files from {read_paths}. Enable delta: {self.config.enable_delta}"
                )
                return

            return read_files

        read_tasks = []

        read_task_fn = create_read_task_fn(self.paths)
        read_task = ReadTask(
            read_task_fn, BlockMetadata(None, None, None, None, self.paths)
        )
        read_tasks.append(read_task)

        return read_tasks

    def supports_distributed_reads(self):
        return False

    def estimate_inmemory_data_size(self):
        return 5 * 1024 * 1024
