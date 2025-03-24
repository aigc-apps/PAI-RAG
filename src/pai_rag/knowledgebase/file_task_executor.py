from typing import Generator
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import PaiNodeParser
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
from pai_rag.knowledgebase.models import (
    FileItem,
    FileOperationType,
    FileProcessResult,
    FileProcessStatus,
)
import traceback
from loguru import logger

from pai_rag.knowledgebase.rag_knowledgebase import knowledgebase_manager
from pai_rag.knowledgebase.rag_knowledgebase_helper import RagKnowledgeBaseHelper

DEFAULT_EMBEDDING_BATCH_SIZE = 1000


class FileTaskExecutor:
    def __init__(
        self,
        data_reader: PaiDataReader,
        node_parser: PaiNodeParser,
        embed_model: PaiEmbedding,
        vector_index: PaiVectorStoreIndex,
    ):
        self.data_reader = data_reader
        self.node_parser = node_parser
        self.embed_model = embed_model
        self.vector_index = vector_index

    def _update(self, task: FileItem) -> Generator[FileProcessResult, None, None]:
        try:
            self._delete(task)
            logger.info(f"Delete file {task.file_name} successfully.")
            yield FileProcessResult(status=FileProcessStatus.Done, message=None)
        except Exception as ex:
            logger.error(
                f"Delete file {task.file_name} failed: {traceback.format_exc()}"
            )
            yield FileProcessResult(status=FileProcessStatus.Failed, message=str(ex))

        for r in self._add_gen(task):
            yield r

    def _add_gen(self, task: FileItem) -> Generator[FileProcessResult, None, None]:
        knowledgebase = knowledgebase_manager.get_knowledgebase(task.knowledgebase)
        yield FileProcessResult(status=FileProcessStatus.Parsing, message=None)
        try:
            docs = self.data_reader.load_data(file_path_or_directory=task.file_name)
            # 对于表格类型，会变成多个文件的，共用同一个id
            for doc in docs:
                doc.id_ = task.task_id

            RagKnowledgeBaseHelper.save_parse_files(knowledgebase.name, docs)
            logger.info(f"Parse file successfully for {task.file_name}")
        except Exception as ex:
            logger.error(
                f"Parse file {task.file_name} failed: {traceback.format_exc()}"
            )
            yield FileProcessResult(status=FileProcessStatus.Failed, message=str(ex))
            return

        yield FileProcessResult(status=FileProcessStatus.Chunking, message=None)
        try:
            chunks = self.node_parser(docs)
            if len(chunks) > 1000:
                logger.warning(
                    f"File {task.file_name} has too many chunks with size {len(chunks)}, skipping save chunks."
                )
            else:
                RagKnowledgeBaseHelper.save_chunk_nodes(
                    knowledgebase.name, chunks, "split"
                )
                logger.info(f"Chunk nodes successfully for file {task.file_name}")
        except Exception as ex:
            logger.error(
                f"Chunk file {task.file_name} failed: {traceback.format_exc()}"
            )
            yield FileProcessResult(status=FileProcessStatus.Failed, message=str(ex))
            return

        yield FileProcessResult(status=FileProcessStatus.Embedding, message=None)
        try:
            embedded_nodes = []
            embedding_batch_size = DEFAULT_EMBEDDING_BATCH_SIZE
            yield FileProcessResult(status=FileProcessStatus.Persisting, message=None)
            for i in range(0, len(chunks), embedding_batch_size):
                batch_chunks = chunks[i : i + embedding_batch_size]
                embedded_batch_nodes = self.embed_model(batch_chunks)
                del batch_chunks
                embedded_nodes.extend(embedded_batch_nodes)
                try:
                    self.vector_index.insert_nodes(nodes=embedded_batch_nodes)
                    logger.info(
                        f"Persist {i + embedding_batch_size} nodes successfully."
                    )
                    del embedded_batch_nodes
                except Exception as ex:
                    logger.error(
                        f"Persist nodes for file {task.file_name} failed: {traceback.format_exc()}"
                    )
                    yield FileProcessResult(
                        status=FileProcessStatus.Failed, message=str(ex)
                    )
                    return
            logger.info(f"Persist nodes successfully for file {task.file_name}")
            logger.info(f"Add file to index succuessfully {task.file_name}.")
            yield FileProcessResult(status=FileProcessStatus.Done, message=None)
            if len(embedded_nodes) > 1000:
                logger.warning(
                    f"File {task.file_name} has too many chunks with size {len(embedded_nodes)}, skipping save embed chunks."
                )
            else:
                RagKnowledgeBaseHelper.save_chunk_nodes(
                    knowledgebase.name, embedded_nodes, "embed"
                )
                logger.info(
                    f"Get nodes embedding successfully for file {task.file_name}"
                )
        except Exception as ex:
            logger.error(
                f"Embedding file {task.file_name} failed: {traceback.format_exc()}"
            )
            yield FileProcessResult(status=FileProcessStatus.Failed, message=str(ex))
            return

    def _delete(self, task: FileItem):
        self.vector_index.delete_ref_doc(ref_doc_id=task.task_id)

    def run(self, task: FileItem) -> Generator[FileProcessResult, None, None]:
        if task.operation == FileOperationType.DELETE:
            try:
                self._delete(task)
                logger.info(f"Delete file {task.file_name} successfully.")
                yield FileProcessResult(status=FileProcessStatus.Done, message=None)
            except Exception as e:
                logger.error(f"Delete file {task.file_name} failed: {e}")
                yield FileProcessResult(status=FileProcessStatus.Failed, message=str(e))
        elif task.operation == FileOperationType.ADD:
            for resp in self._add_gen(task):
                yield resp
        elif task.operation == FileOperationType.UPDATE:
            for resp in self._update(task):
                yield resp
        else:
            raise ValueError(f"Unknown operation {task.operation}.")
