from enum import Enum
from typing import Dict, List, Literal
from pydantic import BaseModel

from pai_rag.app.web.ui_constants import NL2SQL_GENERAL_PROMPTS, SYN_GENERAL_PROMPTS


class DataAnalysisType(str, Enum):
    """Data Analysis types."""

    pandas = "pandas"
    sqlite = "sqlite"
    mysql = "mysql"


class BaseAnalysisConfig(BaseModel):
    """Base class for data analysis config."""

    type: DataAnalysisType
    nl2sql_prompt: str = NL2SQL_GENERAL_PROMPTS
    synthesizer_prompt: str = SYN_GENERAL_PROMPTS


class PandasAnalysisConfig(BaseAnalysisConfig):
    type: Literal[DataAnalysisType.pandas] = DataAnalysisType.pandas
    file_path: str = "./localdata/data_analysis/"


class SqlAnalysisConfig(BaseAnalysisConfig):
    database: str
    tables: List[str] = []
    descriptions: Dict[str, str] = {}
    # offline
    enable_enhanced_description: bool = False
    enable_db_history: bool = False
    enable_db_embedding: bool = False
    # online
    enable_query_preprocessor: bool = False
    enable_db_preretriever: bool = False
    enable_db_selector: bool = False


class SqliteAnalysisConfig(SqlAnalysisConfig):
    type: Literal[DataAnalysisType.sqlite] = DataAnalysisType.sqlite
    db_path: str


class MysqlAnalysisConfig(SqlAnalysisConfig):
    type: Literal[DataAnalysisType.mysql] = DataAnalysisType.mysql
    user: str
    password: str
    host: str
    port: int
