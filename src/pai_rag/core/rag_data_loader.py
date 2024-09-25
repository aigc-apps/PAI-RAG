from typing import Any
from llama_index.core.schema import TransformComponent
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import PaiNodeParser
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
import logging

logger = logging.getLogger(__name__)


class RagDataLoader:
    def __init__(
        self,
        data_reader: PaiDataReader,
        node_parser: PaiNodeParser,
        raptor_processor: TransformComponent = None,
        embed_model: Any = None,
        multi_modal_embed_modal: Any = None,
        vector_index: VectorStoreIndex = None,
    ):
        self._data_reader = data_reader
        self._node_parser = node_parser
        self._raptor_processor = raptor_processor

        self._embed_model = embed_model
        self._multi_modal_embed_model = multi_modal_embed_modal
        self._vector_index = vector_index

    def load_data(
        self,
        file_path_or_directory: str,
        from_oss: bool = False,
        oss_path: str = None,
        filter_pattern: str = None,
        enable_raptor: bool = False,
    ):
        """Load data from a file or directory."""
        documents = self._data_reader.load_data(
            file_path_or_directory=file_path_or_directory,
            filter_pattern=filter_pattern,
            oss_path=oss_path,
            from_oss=from_oss,
        )
        if from_oss:
            logger.info(f"Loaded {len(documents)} documents from {oss_path}")
        else:
            logger.info(
                f"Loaded {len(documents)} documents from {file_path_or_directory}"
            )

        if enable_raptor:
            assert self._raptor_processor is not None, "Raptor processor is not set."
            ingestion_pipeline = IngestionPipeline(
                transformations=[
                    self._node_parser,
                    self._embed_model,
                    self._multi_modal_embed_model,
                    self._raptor_processor,
                ]
            )
        else:
            ingestion_pipeline = IngestionPipeline(
                transformations=[
                    self._node_parser,
                    self._embed_model,
                    self._multi_modal_embed_model,
                ]
            )

        nodes = ingestion_pipeline.run(documents=documents)
        logger.info(
            f"[DataLoader] parsed {len(documents)} documents into {len(nodes)} nodes."
        )

        self._vector_index.insert_nodes(nodes)
        logger.info(f"[DataLoader] Inserted {len(nodes)} nodes.")
        logger.info("[DataLoader] Ingestion Completed!")
