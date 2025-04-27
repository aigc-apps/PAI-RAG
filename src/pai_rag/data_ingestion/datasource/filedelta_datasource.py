import os
from loguru import logger
from ray.data.datasource import ReadTask, FileBasedDatasource
from ray.air.util.tensor_extensions.arrow import pyarrow_table_from_pydict
from ray.data.block import BlockMetadata
from typing import List, Dict
from pai_rag.data_ingestion.constants import DEFAULT_MODIFIED_AT_FIELD, DEFAULT_NODE_SOURCE_FIELD
from pai_rag.data_ingestion.delta.list_delta import list_files_from_langstudio_index, list_files_from_rag_service
from pai_rag.data_ingestion.delta.models import DocItem
from pai_rag.data_ingestion.models.config.datasource import DataSourceConfig
from pai_rag.data_ingestion.models.file.event import FileChangeType, NodeOperationType
from pai_rag.data_ingestion.operators.parser import get_path_resolver



class FileDeltaDatasource(FileBasedDatasource):
    def __init__(
        self,
        config: DataSourceConfig
    ):
        super().__init__(
            paths=config.input_path,
            file_extensions=config.file_extensions,
        )
        self.path_resolver = get_path_resolver()
        oss_path_prefix = self.path_resolver.resolve_source_url(config.input_path)

        # default empty
        self.docs_in_store: Dict[str, DocItem] = {}
        if config.enable_delta and config.target_index:
            logger.info(f"""Reading existing docs from vector store using langstudio index. 
                index_name: {config.target_index}
                version: {config.target_index_version}
            """)
            self.docs_in_store = list_files_from_langstudio_index(
                index_name=config.target_index,
                index_version=config.target_index_version,
                oss_path_prefix=oss_path_prefix,
            )
        elif config.enable_delta and config.rag_api_key and config.rag_endpoint and config.knowledgebase:
            logger.info(f"""Reading existing docs from vector store using langstudio index. 
                rag_endpoint: {config.rag_endpoint}
                knowledgebase: {config.knowledgebase}
                embed_dims: {config.embed_dims}
            """)
            self.docs_in_store = list_files_from_rag_service(
                rag_api_key=config.rag_api_key,
                rag_endpoint=config.rag_endpoint,
                knowledgebase=config.knowledgebase,
                embed_dims=config.embed_dims,
                oss_path_prefix=oss_path_prefix,
            )
        elif config.enable_delta:
            logger.error(
                f"Please specify the information needed to compute delta."
            )
        else:
            logger.info(f"Will not compute delta. This is a full load.")


    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        """
        获取读取任务，返回一个或多个 ReadTask。
        """
        def read_files() -> List[Dict]:
            """
            准备读取任务，返回一个或多个 ReadTask。
            """
            # 获取文件夹中的所有文件信息
            file_info_list = []

            existing_file_set = set()
            for file_path in self._paths():
                file_name = os.path.basename(file_path)
                file_uri = self.path_resolver.resolve_source_url(file_path)
                existing_file_set.add(file_uri)
                doc_item = self.docs_in_store.get(file_uri)
                if doc_item:
                    file_mtime = os.stat(file_path).st_mtime
                    if file_mtime <= doc_item.modified_time:
                        logger.debug(
                            f"Skipping {file_name} as it has not been modified since last indexing."
                        )
                        continue
                    elif file_mtime > doc_item.modified_time:
                        logger.debug(
                            f"{file_name} has been modified since last indexing."
                        )
                        file_info_list.append(
                            {
                                "node_ids": doc_item.node_ids,
                                "file_name": [file_name],
                                "file_path": [file_path],
                                DEFAULT_NODE_SOURCE_FIELD: [file_uri],
                                DEFAULT_MODIFIED_AT_FIELD: [file_mtime],
                                "operation": [NodeOperationType.DELETE],
                                "operation_reason": [FileChangeType.MODIFY]
                            }
                        )
                        file_info_list.append(
                            {
                                "file_name": [file_name],
                                "file_path": [file_path],
                                DEFAULT_NODE_SOURCE_FIELD: [file_uri],
                                DEFAULT_MODIFIED_AT_FIELD: [file_mtime],
                                "operation": [NodeOperationType.ADD],
                                "operation_reason": [FileChangeType.MODIFY]
                            }
                        )
                else:
                    logger.debug(
                        f"{file_name} is newly added."
                    )
                    file_info_list.append(
                        {
                            "file_name": [file_name],
                            "file_path": [file_path],
                            DEFAULT_NODE_SOURCE_FIELD: [file_uri],
                            DEFAULT_MODIFIED_AT_FIELD: [file_mtime],
                            "operation": [NodeOperationType.ADD],
                            "operation_reason": [FileChangeType.ADD]
                        }
                    )

            for file_uri in self.docs_in_store:
                if file_uri not in existing_file_set:
                    logger.info(f"File {file_uri} has been deleted. Will remove from vector store.")
                    file_info_list.append(
                        {
                            "node_ids": doc_item.node_ids,
                            "file_name": [file_name],
                            "file_path": [file_path],
                            DEFAULT_NODE_SOURCE_FIELD: [file_uri],
                            DEFAULT_MODIFIED_AT_FIELD: [file_mtime],
                            "operation": [NodeOperationType.DELETE],
                            "operation_reason": [FileChangeType.DELETE]
                        }
                    )
            
            file_results = [pyarrow_table_from_pydict(value_dict) for value_dict in file_info_list]

            return file_results

        return [ReadTask(read_files, metadata=BlockMetadata(None,None,None,None,None))]
    
    def estimate_inmemory_data_size(self):
        return None
