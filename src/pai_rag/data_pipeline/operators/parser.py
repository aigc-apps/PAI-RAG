from typing import Any, Dict, List
from llama_index.core.schema import TextNode
from pai_rag.data_pipeline.models.config.operator import ParserConfig
from pai_rag.data_pipeline.models.file.event import NodeOperationType
from pai_rag.data_pipeline.operators.base import BaseOperator
from pai_rag.data_pipeline.utils.download_utils import download_models_via_lock
from pai_rag.data_pipeline.utils.node_utils import node_to_metadata_dict_v2
from pai_rag.data_pipeline.constants import (
    DEFAULT_NODE_SOURCE_FIELD,
    DEFAULT_MODIFIED_AT_FIELD,
    DEFAULT_MD5_FIELD,
)
from pai_rag.file.readers.pai.pai_data_reader import (
    BaseDataReaderConfig,
    PaiDataReader,
)
from pai_rag.data_pipeline.utils.path_resolver import (
    MountPathResolver,
    LocalPathResolver,
)
from pai_rag.data_pipeline.ext.langstudio.langstudio_path_resolver import (
    LangStudioPathResolver,
)
from loguru import logger

from pai_rag.utils.file_utils import generate_file_md5, get_modified_time


def get_path_resolver() -> MountPathResolver:
    try:
        langstudio_path_resolver = LangStudioPathResolver.from_env()
        return langstudio_path_resolver
    except Exception as e:
        logger.warning(f"Failed to create LangStudioPathResolver from env: {e}")
        return LocalPathResolver()


class Parser(BaseOperator):
    def __init__(
        self,
        config: ParserConfig,
    ):
        super().__init__(
            name=config.name,
            num_cpus=config.num_cpus,
            num_gpus=config.num_gpus,
            model_dir=config.model_dir,
        )

        download_models_via_lock(
            self.model_dir, "PDF-Extract-Kit-1.0", use_cuda=self.use_cuda()
        )

        self.data_reader_config = BaseDataReaderConfig(
            concat_csv_rows=config.concat_sheet_rows,
            enable_mandatory_ocr=config.enable_pdf_ocr,
            format_sheet_data_to_json=False,
            sheet_column_filters=None,
        )

        self.data_reader = PaiDataReader(
            reader_config=self.data_reader_config,
            image_store=None,
        )
        self.path_resolver = get_path_resolver()
        logger.info(
            f"""Parser operator init finished with following parameters: {config}"""
        )

    def process_delete(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        node_ids = row.get("node_ids", [])
        nodes = []
        for node_id in node_ids:
            node = TextNode(id_=node_id)
            node_dict = node_to_metadata_dict_v2(node)
            node_dict["operation"] = row.get("operation")
            node_dict["operation_reason"] = row.get("operation_reason")
            node_dict["file_name"] = row.get("file_name")
            node_dict["file_path"] = row.get("file_path")
            nodes.append(node_dict)

        return nodes

    def process_add(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        file_path = row.get("file_path")
        nodes = []
        if file_path:
            input_files = [file_path]
            documents = self.data_reader.load_data(file_path_or_directory=input_files)
            if len(documents) == 0:
                logger.warning(f"No data found in the input files: {input_files}")
                return nodes

            file_md5 = generate_file_md5(file_path)
            file_mtime = get_modified_time(file_path)
            file_uri = self.path_resolver.resolve_source_url(file_path)
            for doc in documents:
                doc.metadata[DEFAULT_NODE_SOURCE_FIELD] = file_uri
                doc.metadata[DEFAULT_MD5_FIELD] = file_md5
                doc.metadata[DEFAULT_MODIFIED_AT_FIELD] = file_mtime
                node_dict = node_to_metadata_dict_v2(doc)
                node_dict["operation"] = row.get("operation")
                node_dict["operation_reason"] = row.get("operation_reason")
                nodes.append(node_dict)

            logger.info(f"Parsed {len(nodes)} nodes in the input files: {input_files}")

        else:
            logger.warning("No `file_path` field found in the input entries.")

        return nodes

    def __call__(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        if row.get("operation") == NodeOperationType.DELETE:
            return self.process_delete(row)
        else:
            return self.process_add(row)
