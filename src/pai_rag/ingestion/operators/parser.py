from typing import Any, Dict, List
from llama_index.core.schema import TextNode
from pai_rag.ingestion.models.config.base import ParserConfig
from pai_rag.ingestion.models.file.event import NodeOperationType
from pai_rag.ingestion.operators.base import BaseOperator
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from pai_rag.ingestion.utils.download_utils import download_models_via_lock
from pai_rag.integrations.readers.pai.pai_data_reader import (
    BaseDataReaderConfig,
    PaiDataReader,
)
from loguru import logger


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

        download_models_via_lock(self.model_dir, "PDF-Extract-Kit", use_cuda=self.use_cuda())

        self.data_reader_config = BaseDataReaderConfig(
            concat_csv_rows=config.concat_sheet_rows,
            enable_mandatory_ocr=config.enable_pdf_ocr,
            format_sheet_data_to_json=False,
            sheet_column_filters=None,
        )

        self.data_reader = PaiDataReader(
            reader_config=self.data_reader_config,
            oss_store=None,
        )
        logger.info(
            f"""Parser operator init finished with following parameters: {config}"""
        )

    def process_delete(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        node_ids = row.get("node_ids", [])
        nodes = []
        for node_id in node_ids:
            node = TextNode(node_id=node_id)
            node_dict = node_to_metadata_dict(node)
            node_dict["operation"] = row.get("operation")
            node_dict["operation_reason"] = row.get("operation_reason")
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
            
            for doc in documents:
                node_dict = node_to_metadata_dict(doc)
                node_dict["operation"] = row.get("operation")
                node_dict["operation_reason"] = row.get("operation_reason")
                nodes.append(node_dict)
                
        return nodes

    def __call__(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        if row.get("operation") == NodeOperationType.DELETE:
            return self.process_delete(row)
        else:
            return self.process_add(row)