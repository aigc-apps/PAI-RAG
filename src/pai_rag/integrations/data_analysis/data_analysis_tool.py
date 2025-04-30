import os
from typing import Optional, List, Tuple, Any
from loguru import logger

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.settings import Settings
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.prompts.mixin import PromptMixinType
import llama_index.core.instrumentation as instrument

from pai_rag.integrations.data_analysis.nl2pandas_retriever import PandasQueryRetriever
from pai_rag.integrations.data_analysis.data_analysis_synthesizer import (
    DataAnalysisSynthesizer,
)
from pai_rag.integrations.data_analysis.text2sql.db_connector import (
    MysqlConnector,
    SqliteConnector,
)
from pai_rag.integrations.data_analysis.text2sql.db_info_retriever import (
    SchemaRetriever,
    HistoryRetriever,
    ValueRetriever,
)
from pai_rag.integrations.data_analysis.text2sql.db_loader import DBLoader
from pai_rag.integrations.data_analysis.text2sql.db_query import DBQuery
from pai_rag.integrations.data_analysis.data_analysis_config import (
    BaseAnalysisConfig,
    PandasAnalysisConfig,
    SqlAnalysisConfig,
    MysqlAnalysisConfig,
    SqliteAnalysisConfig,
)
from pai_rag.integrations.data_analysis.text2sql.utils.prompts import (
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT,
)

dispatcher = instrument.get_dispatcher(__name__)


cls_cache = {}


def resolve(cls: Any, cls_key: str, **kwargs):
    cls_key = kwargs.__repr__() + cls_key
    if cls_key not in cls_cache:
        cls_cache[cls_key] = cls(**kwargs)
        instance = cls(**kwargs)
        logger.debug(f"Created new instance with id: {id(instance)}")
    else:
        logger.debug(f"Returning cached instance with id: {id(cls_cache[cls_key])}")
    return cls_cache[cls_key]


def resolve_schema_retriever(
    analysis_config: SqlAnalysisConfig, embed_model: BaseEmbedding
):
    return resolve(
        cls=SchemaRetriever,
        cls_key="schema_retriever",
        db_name=analysis_config.database,
        embed_model=embed_model,
        similarity_top_k=8,
    )


def resolve_history_retriever(
    analysis_config: SqlAnalysisConfig, embed_model: BaseEmbedding
):
    return resolve(
        cls=HistoryRetriever,
        cls_key="history_retriever",
        db_name=analysis_config.database,
        embed_model=embed_model,
        similarity_top_k=5,
    )


def resolve_value_retriever(
    analysis_config: SqlAnalysisConfig, embed_model: BaseEmbedding
):
    return resolve(
        cls=ValueRetriever,
        cls_key="value_retriever",
        db_name=analysis_config.database,
        embed_model=embed_model,
        similarity_top_k=5,
    )


def create_db_connctor(analysis_config: SqlAnalysisConfig):
    if isinstance(analysis_config, MysqlAnalysisConfig):
        return MysqlConnector(db_config=analysis_config)
    elif isinstance(analysis_config, SqliteAnalysisConfig):
        return SqliteConnector(db_config=analysis_config)
    else:
        raise ValueError(f"Unknown sql analysis config: {analysis_config}.")


def create_query_retriever(
    analysis_config: BaseAnalysisConfig,
    sql_database: SQLDatabase,
    llm: LLM,
    embed_model: BaseEmbedding,
):
    if isinstance(analysis_config, PandasAnalysisConfig):
        return PandasQueryRetriever.from_config(
            pandas_config=analysis_config,
            llm=llm,
        )
    elif isinstance(analysis_config, SqlAnalysisConfig):
        return DBQuery(
            db_config=analysis_config,
            sql_database=sql_database,
            embed_model=embed_model,
            schema_retriever=resolve_schema_retriever(analysis_config, embed_model),
            history_retriever=resolve_history_retriever(analysis_config, embed_model),
            value_retriever=resolve_value_retriever(analysis_config, embed_model),
            llm=llm,
        )
    else:
        raise ValueError(f"Unknown sql analysis config: {analysis_config}.")


class DataAnalysisConnector:
    """
    Used for db connection
    """

    def __init__(
        self,
        analysis_config: BaseAnalysisConfig,
    ) -> None:
        self._analysis_config = analysis_config
        if isinstance(analysis_config, PandasAnalysisConfig):
            self._db_connector = None
        elif isinstance(analysis_config, SqlAnalysisConfig):
            self._db_connector = create_db_connctor(analysis_config)
        else:
            raise ValueError(f"Unknown analysis config: {analysis_config}.")

    def connect(self):
        if isinstance(self._analysis_config, PandasAnalysisConfig):
            return
        elif isinstance(self._analysis_config, SqlAnalysisConfig):
            return self._db_connector.connect()
        else:
            raise ValueError(f"Unknown analysis config: {self._analysis_config}.")


class DataAnalysisLoader:
    """
    Used for db info collection and index creation.
    """

    def __init__(
        self,
        analysis_config: SqlAnalysisConfig,
        sql_database: SQLDatabase,
        embed_model: BaseEmbedding,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._sql_database = sql_database
        self._db_loader = DBLoader(
            db_config=analysis_config,
            sql_database=sql_database,
            embed_model=embed_model,
            schema_retriever=resolve_schema_retriever(analysis_config, embed_model),
            history_retriever=resolve_history_retriever(analysis_config, embed_model),
            value_retriever=resolve_value_retriever(analysis_config, embed_model),
            llm=llm,
        )

    def load_db_info(self):
        return self._db_loader.load_db_info()

    async def aload_db_info(self):
        return await self._db_loader.aload_db_info()


class DataAnalysisQuery(BaseQueryEngine):
    """
    Used for db or excel/csv file Data Query
    """

    def __init__(
        self,
        analysis_config: BaseAnalysisConfig,
        sql_database: SQLDatabase,
        embed_model: BaseEmbedding,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Initialize params."""
        self._llm = llm or Settings.llm
        self._embed_model = embed_model
        self._sql_database = sql_database
        self._query_retriever = create_query_retriever(
            analysis_config=analysis_config,
            sql_database=self._sql_database,
            llm=self._llm,
            embed_model=self._embed_model,
        )
        self._synthesizer = DataAnalysisSynthesizer(
            llm=self._llm,
            response_synthesis_prompt=PromptTemplate(analysis_config.synthesizer_prompt)
            or DEFAULT_RESPONSE_SYNTHESIS_PROMPT,
        )
        super().__init__(callback_manager=callback_manager or Settings.callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._query_retriever.retrieve(query_bundle)
        if isinstance(nodes, Tuple):
            return nodes[0], nodes[1]
        else:
            return nodes, ""

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._query_retriever.aretrieve(query_bundle)
        if isinstance(nodes, Tuple):
            return nodes[0], nodes[1]
        else:
            return nodes, ""

    def synthesize(
        self,
        query_bundle: QueryBundle,
        description: str,
        nodes: List[NodeWithScore],
        streaming: bool = False,
    ) -> RESPONSE_TYPE:
        return self._synthesizer.synthesize(
            query=query_bundle,
            description=description,
            nodes=nodes,
            streaming=streaming,
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        description: str,
        nodes: List[NodeWithScore],
        streaming: bool = False,
    ) -> RESPONSE_TYPE:
        return await self._synthesizer.asynthesize(
            query=query_bundle,
            description=description,
            nodes=nodes,
            streaming=query_bundle.stream,
        )

    @dispatcher.span
    def _query(
        self,
        query_bundle: QueryBundle,
        streaming: bool = False,
    ) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes, description = self.retrieve(query_bundle)
            response = self._synthesizer.synthesize(
                query=query_bundle,
                description=description,
                nodes=nodes,
                streaming=streaming,
            )
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    @dispatcher.span
    async def _aquery(
        self, query_bundle: QueryBundle, streaming: bool = False
    ) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes, description = await self.aretrieve(query_bundle)
            response = await self._synthesizer.asynthesize(
                query=query_bundle,
                description=description,
                nodes=nodes,
                streaming=query_bundle.stream,
            )
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        nodes, description = await self.aretrieve(query_bundle)
        response = await self._synthesizer.asynthesize(
            query=query_bundle,
            description=description,
            nodes=nodes,
            streaming=query_bundle.stream,
        )

        return response

    async def astream_query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        nodes, description = await self.aretrieve(query_bundle)
        stream_response = await self._synthesizer.asynthesize(
            query=query_bundle, description=description, nodes=nodes, streaming=True
        )

        return stream_response
