from typing import Optional, Any
from loguru import logger

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.settings import Settings
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.prompts.mixin import PromptMixinType
import llama_index.core.instrumentation as instrument

from pairag.integrations.data_analysis.nl2pandas_retriever import PandasQueryRetriever
from pairag.integrations.data_analysis.text2sql.db_connector import (
    MysqlConnector,
    SqliteConnector,
)
from pairag.integrations.data_analysis.text2sql.db_info_retriever import (
    SchemaRetriever,
    HistoryRetriever,
    ValueRetriever,
)
from pairag.integrations.data_analysis.text2sql.db_loader import DBLoader
from pairag.integrations.data_analysis.text2sql.db_query import DBQuery
from pairag.integrations.data_analysis.data_analysis_config import (
    BaseAnalysisConfig,
    PandasAnalysisConfig,
    SqlAnalysisConfig,
    MysqlAnalysisConfig,
    SqliteAnalysisConfig,
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


class SqlRetriever(BaseRetriever):
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
        super().__init__(callback_manager=callback_manager or Settings.callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    async def _aretrieve(self, query_bundle):
        return await self._query_retriever.aretrieve(query_bundle)

    def _retrieve(self, query_bundle):
        raise NotImplementedError
