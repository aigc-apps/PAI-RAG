import os
from typing import Optional, List, Tuple
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
from llama_index.core.utilities.sql_wrapper import SQLDatabase
import llama_index.core.instrumentation as instrument

from pai_rag.integrations.data_analysis.data_analysis_config import (
    BaseAnalysisConfig,
    PandasAnalysisConfig,
    SqlAnalysisConfig,
)
from pai_rag.integrations.data_analysis.nl2pandas_retriever import PandasQueryRetriever
from pai_rag.integrations.data_analysis.data_analysis_synthesizer import (
    DataAnalysisSynthesizer,
)
from pai_rag.integrations.data_analysis.nl2sql.db_utils.constants import (
    DESCRIPTION_STORAGE_PATH,
    HISTORY_STORAGE_PATH,
    VALUE_STORAGE_PATH,
)
from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT,
)
from pai_rag.integrations.data_analysis.nl2sql.db_connector import DBConnector
from pai_rag.integrations.data_analysis.nl2sql.db_loader import DBLoader
from pai_rag.integrations.data_analysis.nl2sql.db_query import DBQuery
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex
from pai_rag.integrations.index.pai.vector_store_config import FaissVectorStoreConfig

dispatcher = instrument.get_dispatcher(__name__)


if os.path.exists("./model_repository/bge-large-zh-v1.5"):
    embed_model_bge_large = HuggingFaceEmbedding(
        model_name="./model_repository/bge-large-zh-v1.5"
    )
else:
    embed_model_bge_large = None


def create_db_connctor(analysis_config: BaseAnalysisConfig):
    if isinstance(analysis_config, SqlAnalysisConfig):
        return DBConnector.from_config(sql_config=analysis_config)
    else:
        raise ValueError(f"Unknown sql analysis config: {analysis_config}.")


def create_db_indexes(analysis_config: SqlAnalysisConfig, embed_model: BaseEmbedding):
    db_name = analysis_config.database
    description_persist_path = os.path.join(DESCRIPTION_STORAGE_PATH, db_name)
    history_persist_path = os.path.join(HISTORY_STORAGE_PATH, db_name)
    value_persist_path = os.path.join(VALUE_STORAGE_PATH, db_name)

    description_vector_store_config = FaissVectorStoreConfig(
        persist_path=description_persist_path,
    )
    history_vector_store_config = FaissVectorStoreConfig(
        persist_path=history_persist_path,
    )
    value_vector_store_config = FaissVectorStoreConfig(
        persist_path=value_persist_path,
    )
    description_index = PaiVectorStoreIndex(
        vector_store_config=description_vector_store_config, embed_model=embed_model
    )
    history_index = PaiVectorStoreIndex(
        vector_store_config=history_vector_store_config, embed_model=embed_model
    )
    value_index = PaiVectorStoreIndex(
        vector_store_config=value_vector_store_config, embed_model=embed_model
    )
    return [description_index, history_index, value_index]


def create_db_loader(
    analysis_config: BaseAnalysisConfig,
    sql_database: SQLDatabase,
    # embed_model: BaseEmbedding,
    # indexes: List[PaiVectorStoreIndex],
    llm: LLM,
):
    if isinstance(analysis_config, SqlAnalysisConfig):
        print("analysis_config:", analysis_config)
        embed_model = embed_model_bge_large
        indexes = create_db_indexes(analysis_config, embed_model)
        # print("indexes:", indexes)
        return DBLoader.from_config(
            sql_config=analysis_config,
            sql_database=sql_database,
            embed_model=embed_model,
            index=indexes,
            llm=llm,
        )
    else:
        raise ValueError(f"Unknown sql analysis config: {analysis_config}.")


def create_retriever(
    analysis_config: BaseAnalysisConfig,
    sql_database: SQLDatabase,
    # indexes: List[PaiVectorStoreIndex],
    llm: LLM,
    # embed_model: BaseEmbedding,
):
    if isinstance(analysis_config, PandasAnalysisConfig):
        return PandasQueryRetriever.from_config(
            pandas_config=analysis_config,
            llm=llm,
        )
    elif isinstance(analysis_config, SqlAnalysisConfig):
        embed_model = embed_model_bge_large
        indexes = create_db_indexes(analysis_config, embed_model)
        return DBQuery.from_config(
            sql_config=analysis_config,
            sql_database=sql_database,
            embed_model=embed_model,
            index=indexes,
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

    def connect_db(self):
        if isinstance(self._analysis_config, PandasAnalysisConfig):
            return
        elif isinstance(self._analysis_config, SqlAnalysisConfig):
            return self._db_connector.connect()
        else:
            raise ValueError(f"Unknown analysis config: {self._analysis_config}.")


class DataAnalysisLoader:
    """
    Used for db connection, description and embedding
    """

    def __init__(
        self,
        analysis_config: BaseAnalysisConfig,
        sql_database: SQLDatabase,
        # embed_model: BaseEmbedding,
        # indexes: List[PaiVectorStoreIndex],
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._llm = llm or Settings.llm
        # self._embed_model = embed_model
        self._sql_database = sql_database
        self._db_loader = create_db_loader(
            analysis_config=analysis_config,
            sql_database=self._sql_database,
            # embed_model=embed_model,
            # indexes=indexes,
            llm=self._llm,
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
        # indexes: List[PaiVectorStoreIndex],
        llm: Optional[LLM] = None,
        # embed_model: Optional[BaseEmbedding] = embed_model_bge_large,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Initialize params."""
        self._llm = llm or Settings.llm
        self._sql_database = sql_database
        self._retriever = create_retriever(
            analysis_config=analysis_config,
            sql_database=self._sql_database,
            # indexes=indexes,
            llm=self._llm,
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
        if isinstance(nodes, Tuple):
            return nodes[0], nodes[1]
        else:
            return nodes, ""

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)
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
            streaming=streaming,
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
                streaming=streaming,
            )
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def astream_query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        # streaming = self._synthesizer._streaming
        # self._synthesizer._streaming = True

        nodes, description = await self.aretrieve(query_bundle)

        stream_response = await self._synthesizer.asynthesize(
            query=query_bundle, description=description, nodes=nodes, streaming=True
        )
        # self._synthesizer._streaming = streaming

        return stream_response
