from typing import Any
from llama_index.core.schema import TransformComponent
from llama_index.core.indices import VectorStoreIndex
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import PaiNodeParser
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
from pai_rag.core.rag_knowledgebase_manager import RagKnowledgeBaseManager
from pai_rag.core.rag_job_manager import upload_job_manager
from loguru import logger


class RagDataLoader:
    def __init__(
        self,
        data_reader: PaiDataReader,
        node_parser: PaiNodeParser,
        raptor_processor: TransformComponent = None,
        embed_model: Any = None,
        multimodal_embed_model: Any = None,
        vector_index: VectorStoreIndex = None,
    ):
        self._data_reader = data_reader
        self._node_parser = node_parser
        self._raptor_processor = raptor_processor

        self._embed_model = embed_model
        self._multimodal_embed_model = multimodal_embed_model
        self._vector_index = vector_index

    def load_data(
        self,
        file_path_or_directory: str,
        from_oss: bool = False,
        oss_path: str = None,
        filter_pattern: str = None,
        enable_raptor: bool = False,
        index_name: str = None,
        task_id: str = None,
    ):
        _knowledgebase_manager = RagKnowledgeBaseManager(index_name=index_name)
        """Load data from a file or directory."""
        # parse input files into documents
        upload_job_manager.track_job(
            task_id,
            index_name,
            file_path_or_directory,
            stage="parse",
            status="processing",
        )
        documents = self._data_reader.load_data(
            file_path_or_directory=file_path_or_directory,
            filter_pattern=filter_pattern,
            oss_path=oss_path,
            from_oss=from_oss,
        )
        _knowledgebase_manager.save_parse_files(documents)
        upload_job_manager.track_job(
            task_id,
            index_name,
            file_path_or_directory,
            stage="parse",
            status="completed",
        )
        if from_oss:
            logger.info(f"Loaded {len(documents)} documents from {oss_path}")
        else:
            logger.info(
                f"Loaded {len(documents)} documents from {file_path_or_directory}"
            )

        # split documents into nodes
        upload_job_manager.track_job(
            task_id,
            index_name,
            file_path_or_directory,
            stage="split",
            status="processing",
        )
        splitted_nodes = self._node_parser(documents)
        _knowledgebase_manager.save_chunk_nodes(splitted_nodes, "split")
        upload_job_manager.track_job(
            task_id,
            index_name,
            file_path_or_directory,
            stage="split",
            status="completed",
        )

        # embed nodes
        upload_job_manager.track_job(
            task_id,
            index_name,
            file_path_or_directory,
            stage="embed",
            status="processing",
        )
        embedded_nodes = self._embed_model(splitted_nodes)
        if self._multimodal_embed_model is not None:
            embedded_nodes = self._multimodal_embed_model(embedded_nodes)
        _knowledgebase_manager.save_chunk_nodes(embedded_nodes, "embed")
        upload_job_manager.track_job(
            task_id,
            index_name,
            file_path_or_directory,
            stage="embed",
            status="completed",
        )

        # if enable_raptor:
        #     assert self._raptor_processor is not None, "Raptor processor is not set."
        #     raptor_node = self._raptor_processor(embedded_nodes)
        #     save_chunk_nodes(index_name, embedded_nodes, "raptor")

        logger.info(
            f"[DataLoader] parsed {len(documents)} documents into {len(embedded_nodes)} nodes."
        )

        self._vector_index.insert_nodes(embedded_nodes)
        logger.info(f"[DataLoader] Inserted {len(embedded_nodes)} nodes.")
        logger.info("[DataLoader] Ingestion Completed!")
