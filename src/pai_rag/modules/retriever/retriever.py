"""retriever factory, used to generate retriever instance based on customer config and index"""

import logging
from typing import Dict, List, Any

# from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.indices.list.base import SummaryIndex

from pai_rag.integrations.index.multi_modal_index import MyMultiModalVectorStoreIndex
from pai_rag.integrations.retrievers.bm25 import BM25Retriever
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.utils.prompt_template import (
    QUERY_GEN_PROMPT,
)
from pai_rag.modules.retriever.my_vector_index_retriever import MyVectorIndexRetriever
from pai_rag.integrations.retrievers.fusion_retriever import MyQueryFusionRetriever


logger = logging.getLogger(__name__)


class RetrieverModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["IndexModule", "BM25IndexModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        index = new_params["IndexModule"]
        bm25_index = new_params["BM25IndexModule"]

        similarity_top_k = config.get("similarity_top_k", 5)
        image_similarity_top_k = config.get("image_similarity_top_k", 2)

        retrieval_mode = config.get("retrieval_mode", "hybrid").lower()
        need_image = config.get("need_image", False)
        print("index.vector_index", index.vector_index, type(index.vector_index))
        if isinstance(index.vector_index, MyMultiModalVectorStoreIndex):
            return index.vector_index.as_retriever(
                need_image=need_image,
                similarity_top_k=similarity_top_k,
                image_similarity_top_k=image_similarity_top_k,
            )
        # Special handle elastic search
        elif index.vectordb_type == "milvus":
            if retrieval_mode == "embedding":
                query_mode = VectorStoreQueryMode.DEFAULT
            elif retrieval_mode == "keyword":
                query_mode = VectorStoreQueryMode.TEXT_SEARCH
            else:
                query_mode = VectorStoreQueryMode.HYBRID

            return MyVectorIndexRetriever(
                index=index.vector_index,
                similarity_top_k=similarity_top_k,
                vector_store_query_mode=query_mode,
            )
        elif index.vectordb_type == "elasticsearch":
            if retrieval_mode != "hybrid":
                if retrieval_mode == "embedding":
                    query_mode = VectorStoreQueryMode.DEFAULT
                elif retrieval_mode == "keyword":
                    query_mode = VectorStoreQueryMode.TEXT_SEARCH
                return MyVectorIndexRetriever(
                    index=index.vector_index,
                    similarity_top_k=similarity_top_k,
                    vector_store_query_mode=query_mode,
                )
            else:
                vector_retriever = MyVectorIndexRetriever(
                    index=index.vector_index,
                    similarity_top_k=similarity_top_k,
                    vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
                )
                bm25_retriever = MyVectorIndexRetriever(
                    index=index.vector_index,
                    similarity_top_k=similarity_top_k,
                    vector_store_query_mode=VectorStoreQueryMode.TEXT_SEARCH,
                )
        elif index.vectordb_type == "postgresql":
            if retrieval_mode == "embedding":
                query_mode = VectorStoreQueryMode.DEFAULT
            elif retrieval_mode == "keyword":
                query_mode = VectorStoreQueryMode.TEXT_SEARCH
            else:
                query_mode = VectorStoreQueryMode.HYBRID
            return MyVectorIndexRetriever(
                index=index.vector_index,
                similarity_top_k=similarity_top_k,
                vector_store_query_mode=query_mode,
            )
        else:
            vector_retriever = MyVectorIndexRetriever(
                index=index.vector_index, similarity_top_k=similarity_top_k
            )
            bm25_retriever = BM25Retriever.from_defaults(
                bm25_index=bm25_index,
                similarity_top_k=similarity_top_k,
            )

        if retrieval_mode == "embedding":
            logger.info(f"MyVectorIndexRetriever used with top_k {similarity_top_k}.")
            return vector_retriever

        elif retrieval_mode == "keyword":
            logger.info(f"BM25Retriever used with top_k {similarity_top_k}.")
            return bm25_retriever

        else:
            num_queries_gen = config.get("query_rewrite_n", 3)
            if config["retrieval_mode"] == "hybrid":
                fusion_retriever = MyQueryFusionRetriever(
                    [vector_retriever, bm25_retriever],
                    similarity_top_k=similarity_top_k,
                    num_queries=num_queries_gen,  # set this to 1 to disable query generation
                    use_async=True,
                    verbose=True,
                    query_gen_prompt=QUERY_GEN_PROMPT,
                )
                logger.info(f"FusionRetriever used with top_k {similarity_top_k}.")
                return fusion_retriever

            elif config["retrieval_mode"] == "router":
                nodes = list(index.vector_index.docstore.docs.values())
                summary_index = SummaryIndex(nodes)
                list_retriever = summary_index.as_retriever(
                    retriever_mode="embedding", similarity_top_k=10
                )  # can be 'default' as well for all nodes retrieval

                list_tool = RetrieverTool.from_defaults(
                    retriever=list_retriever,
                    description=("适用于总结类型的查询问题。通常伴随'总结','归纳'等关键词，一般情况下不使用。"),
                )
                vector_tool = RetrieverTool.from_defaults(
                    retriever=fusion_retriever,
                    description=("适用于一般的查询问题，首选。"),
                )

                router_retriever = RouterRetriever(
                    selector=LLMSingleSelector.from_defaults(),
                    retriever_tools=[
                        list_tool,
                        vector_tool,
                    ],
                )

                logger.info("RouterRetriever used.")
                return router_retriever

            else:
                raise ValueError("Not supported retrieval type.")
