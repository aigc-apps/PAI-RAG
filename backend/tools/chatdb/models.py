from pydantic import BaseModel
from enum import Enum


class DbDialect(str, Enum):
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"


class DbConnection(BaseModel):
    dialect: DbDialect = DbDialect.MYSQL
    db_name: str = None
    user_name: str = None
    password: str = None
    port: int = None
    host: str = None
