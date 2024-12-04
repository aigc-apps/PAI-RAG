from typing import Optional, List, Tuple

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
        nodes = self._retriever._retrieve_sql(query_bundle)
        if isinstance(nodes, Tuple):
            return nodes[0], nodes[1]
        else:
            return nodes, ""

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever._aretrieve_sql(query_bundle)
        if isinstance(nodes, Tuple):
            return nodes[0], nodes[1]
        else:
            return nodes, ""

    def synthesize(
        self,
        query_bundle: QueryBundle,
        db_schema: str,
        nodes: List[NodeWithScore],
    ) -> RESPONSE_TYPE:
        return self._synthesizer.synthesize(
            query=query_bundle,
            db_schema=db_schema,
            nodes=nodes,
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        db_schema: str,
        nodes: List[NodeWithScore],
    ) -> RESPONSE_TYPE:
        return await self._synthesizer.asynthesize(
            query=query_bundle,
            db_schema=db_schema,
            nodes=nodes,
        )

    @dispatcher.span
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes, db_schema = self.retrieve(query_bundle)
            response = self._synthesizer.synthesize(
                query=query_bundle,
                db_schema=db_schema,
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
            nodes, db_schema = await self.aretrieve(query_bundle)
            response = await self._synthesizer.asynthesize(
                query=query_bundle,
                db_schema=db_schema,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def astream_query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        streaming = self._synthesizer._streaming
        self._synthesizer._streaming = True

        nodes, db_schema = await self.aretrieve(query_bundle)

        stream_response = await self._synthesizer.asynthesize(
            query=query_bundle, db_schema=db_schema, nodes=nodes
        )
        self._synthesizer._streaming = streaming

        return stream_response
