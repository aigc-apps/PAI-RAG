from typing import Optional, Any

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import Settings
from llama_index.core.schema import QueryBundle


# TODO
class DBPreRetriever:
    """
    基于自然语言问题、关键词等进行预检索，承接db_descriptor，缩小提供的summary范围
    适用于多表多列，当db_descriptor的输出(token)较大时采用，如果很少，直接将全量数据喂入selector

    将数据库描述转化为嵌入向量
    计算输入查询与所有数据库模式描述之间的相似度，返回最相关的描述
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        db_description_index: Optional[Any] = None,
        db_history_index: Optional[Any] = None,
        db_value_index: Optional[Any] = None,
    ) -> None:
        self._llm = llm or Settings.llm
        self._embed_model = embed_model or Settings.embed_model
        self._db_description_index = db_description_index
        self._db_history_index = db_history_index
        self._db_value_index = db_value_index

    def _retrieval_filter(self, retrieved_output) -> str:
        # 根据retrieve的结果缩小db_description中table_column_info以及db_query_history的内容
        pass

    def retrieve(self, nl_query: QueryBundle, *args) -> str:
        # 把summary中的列描述切chunk（一列一条chunk）通过hybrid retrieve的方式，筛选chunk，简化summary，作为pre_retrieved_summary_output
        retrieved_description_output = self._db_description_index.retrieve(nl_query)
        retrieved_history_output = self._db_history_index.retrieve(nl_query)
        retrieved_value_output = self._db_value_index.retrieve(nl_query)

        retrieved_description_str = self._retrieval_filter(retrieved_description_output)
        retrieved_history_str = self._retrieval_filter(retrieved_history_output)
        retrieved_value_str = self._retrieval_filter(retrieved_value_output)

        return retrieved_description_str, retrieved_history_str, retrieved_value_str

    # def embed_index(self, ):
    #     """
    #     为数据库描述生成嵌入向量。

    #     :param schema_descriptions: 一个字典，键是模式标识符（如表名+列名），值是描述字符串。
    #     """

    # def retrieve(self, nlp_query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    #     """
    #     对输入的自然语言查询进行预检索，返回最相关的模式描述。

    #     :param nlp_query: 自然语言查询字符串。
    #     :param top_k: 返回前K个最相关的模式描述。
    #     :return: 一个列表，包含元组形式的模式标识符和相似度分数。
    #     """
