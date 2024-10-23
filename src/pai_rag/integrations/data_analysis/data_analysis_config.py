from enum import Enum
from typing import Dict, List, Literal
from pydantic import BaseModel

from pai_rag.app.web.ui_constants import DA_GENERAL_PROMPTS, SYN_GENERAL_PROMPTS


class DataAnalysisType(str, Enum):
    """Data Analysis types."""

    pandas = "pandas"
    sqlite = "sqlite"
    mysql = "mysql"


class BaseAnalysisConfig(BaseModel):
    """Base class for data analysis config."""

    type: DataAnalysisType
    nl2sql_prompt: str = DA_GENERAL_PROMPTS
    synthesizer_prompt: str = SYN_GENERAL_PROMPTS


class PandasAnalysisConfig(BaseAnalysisConfig):
    type: Literal[DataAnalysisType.pandas] = DataAnalysisType.pandas
    file_path: str = "./localdata/data_analysis/"


class SqlAnalysisConfig(BaseAnalysisConfig):
    database: str
    tables: List[str] = []
    descriptions: Dict[str, str] = {}


class SqliteAnalysisConfig(SqlAnalysisConfig):
    type: Literal[DataAnalysisType.sqlite] = DataAnalysisType.sqlite
    db_path: str


class MysqlAnalysisConfig(SqlAnalysisConfig):
    type: Literal[DataAnalysisType.mysql] = DataAnalysisType.mysql
    user: str
    password: str
    host: str
    port: str = "3306"
