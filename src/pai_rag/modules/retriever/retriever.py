"""retriever factory, used to generate retriever instance based on customer config and index"""

import logging
from typing import Dict, List, Any

import jieba
from nltk.corpus import stopwords
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode

# from llama_index.retrievers.bm25 import BM25Retriever
from pai_rag.integrations.retrievers.bm25 import BM25Retriever
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.utils.prompt_template import QUERY_GEN_PROMPT
from pai_rag.modules.retriever.my_elasticsearch_store import MyElasticsearchStore
from pai_rag.modules.retriever.my_vector_index_retriever import MyVectorIndexRetriever

logger = logging.getLogger(__name__)

stopword_list = stopwords.words("chinese") + stopwords.words("english")


## PUT in utils file and add stopword in TRIE structure.
def jieba_tokenize(text: str) -> List[str]:
    tokens = []
    for w in jieba.lcut(text):
        token = w.lower()
        if token not in stopword_list:
            tokens.append(token)

    return tokens


class RetrieverModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["IndexModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        vector_index = new_params["IndexModule"]

        similarity_top_k = config.get("similarity_top_k", 5)

        retrieval_mode = config.get("retrieval_mode", "hybrid").lower()

        # Special handle elastic search
        if isinstance(vector_index.storage_context.vector_store, MyElasticsearchStore):
            if retrieval_mode == "embedding":
                query_mode = VectorStoreQueryMode.DEFAULT
            elif retrieval_mode == "keyword":
                query_mode = VectorStoreQueryMode.TEXT_SEARCH
            else:
                query_mode = VectorStoreQueryMode.HYBRID

            return MyVectorIndexRetriever(
                index=vector_index,
                similarity_top_k=similarity_top_k,
                vector_store_query_mode=query_mode,
            )

        vector_retriever = MyVectorIndexRetriever(
            index=vector_index, similarity_top_k=similarity_top_k
        )
        # keyword
        bm25_retriever = BM25Retriever.from_defaults(
            index=vector_index,
            similarity_top_k=similarity_top_k,
            tokenizer=jieba_tokenize,
        )

        if retrieval_mode == "embedding":
            logger.info(f"MyVectorIndexRetriever used with top_k {similarity_top_k}.")
            return vector_retriever

        elif retrieval_mode == "keyword":
            logger.info(f"BM25Retriever used with top_k {similarity_top_k}.")
            return bm25_retriever

        else:
            vector_weight = config.get("vector_weight", 0.5)
            keyword_weight = config.get("BM25_weight", 0.5)
            fusion_mode = config.get("fusion_mode", "reciprocal_rerank")
            num_queries_gen = config.get("query_rewrite_n", 3)

            fusion_retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=similarity_top_k,
                num_queries=num_queries_gen,  # set this to 1 to disable query generation
                mode=fusion_mode,
                use_async=True,
                verbose=True,
                query_gen_prompt=QUERY_GEN_PROMPT,
                retriever_weights=[vector_weight, keyword_weight],
            )

            if config["retrieval_mode"] == "hybrid":
                logger.info(f"FusionRetriever used with top_k {similarity_top_k}.")
                return fusion_retriever

            elif config["retrieval_mode"] == "router":
                nodes = list(vector_index.docstore.docs.values())
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
