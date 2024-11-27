from typing import Optional, List

from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.settings import Settings
import llama_index.core.instrumentation as instrument

from sqlalchemy import text

from pai_rag.integrations.data_analysis.nl2sql_retriever import MyNLSQLRetriever
from pai_rag.integrations.data_analysis.data_analysis_config import (
    BaseAnalysisConfig,
    PandasAnalysisConfig,
    SqlAnalysisConfig,
)
from pai_rag.integrations.data_analysis.nl2pandas_retriever import PandasQueryRetriever
from pai_rag.integrations.data_analysis.data_analysis_synthesizer import (
    DataAnalysisSynthesizer,
)

dispatcher = instrument.get_dispatcher(__name__)

DEFAULT_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    "Given an input question, synthesize a response in Chinese from the query results.\n"
    "Query: {query_str}\n\n"
    "SQL or Python Code Instructions (optional):\n{query_code_instruction}\n\n"
    "Code Query Output: {query_output}\n\n"
    "Response: "
)


def create_retriever(
    analysis_config: BaseAnalysisConfig, llm: LLM, embed_model: BaseEmbedding
):
    if isinstance(analysis_config, PandasAnalysisConfig):
        return PandasQueryRetriever.from_config(
            pandas_config=analysis_config,
            llm=llm,
        )
    elif isinstance(analysis_config, SqlAnalysisConfig):
        return MyNLSQLRetriever.from_config(
            sql_config=analysis_config,
            llm=llm,
            embed_model=embed_model,
        )
    else:
        raise ValueError(f"Unknown sql analysis config: {analysis_config}.")


class DataAnalysisTool(BaseQueryEngine):
    """
    Used for db or excel/csv file Data Analysis
    """

    def __init__(
        self,
        analysis_config: BaseAnalysisConfig,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Initialize params."""
        self._llm = llm or Settings.llm
        self._embed_model = embed_model or Settings.embed_model
        self._retriever = create_retriever(
            analysis_config=analysis_config,
            llm=self._llm,
            embed_model=self._embed_model,
        )
        self._synthesizer = DataAnalysisSynthesizer(
            llm=self._llm,
            response_synthesis_prompt=PromptTemplate(analysis_config.synthesizer_prompt)
            or DEFAULT_RESPONSE_SYNTHESIS_PROMPT,
        )
        super().__init__(callback_manager=callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)
        return nodes

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)
        return nodes

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
    ) -> RESPONSE_TYPE:
        return self._synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
    ) -> RESPONSE_TYPE:
        return await self._synthesizer.asynthesize(
            query=query_bundle,
            nodes=nodes,
        )

    @dispatcher.span
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = self.retrieve(query_bundle)
            response = self._synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    @dispatcher.span
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = await self.aretrieve(query_bundle)
            response = await self._synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def astream_query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        streaming = self._synthesizer._streaming
        self._synthesizer._streaming = True

        nodes = await self.aretrieve(query_bundle)

        stream_response = await self._synthesizer.asynthesize(
            query=query_bundle, nodes=nodes
        )
        self._synthesizer._streaming = streaming

        return stream_response


    def sql_query(self, input_list: List) -> List[dict]:
        """Query the material database directly."""
        table_name = self._retriever._tables[0]
        print("table:", table_name)
        columns =  [item["name"] for item in self._retriever._sql_database.get_table_columns(table_name)]
        print("columns:", columns)
        # 使用字符串格式化生成值列表
        value_list = ", ".join(f""" "{code}" """.strip() for code in input_list)
        # 构建 SQL 查询
        sql = f"SELECT * FROM material_data WHERE 物料编码 IN ({value_list})"
        print("sql:", sql)
        try:
            with self._retriever._sql_database.engine.connect() as connection:
                result = connection.execution_options(timeout=60).execute(text(sql))
                query_results = result.fetchall()
            result_json = [dict(zip(columns, sublist)) for sublist in query_results]
            return result_json
        except NotImplementedError as error:
            raise NotImplementedError(f"SQL execution not implemented: {error}") from error
        except Exception as error:
            raise Exception(f"Unexpected error during SQL execution: {error}") from error

