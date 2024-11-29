import os
from typing import List
from pai_rag.utils.constants import DEFAULT_MODEL_DIR

from loguru import logger

MODEL_NAME = "bge-m3"


class BGEM3SparseEmbeddingFunction:
    def __init__(self, model_name_or_path: str = None) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel

            self.model = BGEM3FlagModel(
                model_name_or_path=model_name_or_path
                or os.path.join(DEFAULT_MODEL_DIR, MODEL_NAME),
                use_fp16=False,
            )
        except Exception:
            error_info = (
                "Cannot import BGEM3FlagModel from FlagEmbedding. It seems it is not installed. "
                "Please install it using:\n"
                "pip install FlagEmbedding\n",
                "error_info",
            )

            logger.error(error_info)
            raise

    def encode_queries(self, queries: List[str]):
        outputs = self.model.encode(
            queries, return_dense=False, return_sparse=True, return_colbert_vecs=False
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def encode_documents(self, documents: List[str]):
        outputs = self.model.encode(
            documents, return_dense=False, return_sparse=True, return_colbert_vecs=False
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def _to_standard_dict(self, raw_output):
        result = {}
        for k in raw_output:
            result[int(k)] = raw_output[k]
        return result
