import dotenv
dotenv.load_dotenv()

from db.sqlite_store import init_sqlite_store
import os
from urllib.parse import quote_plus
from logging.config import fileConfig
from loguru import logger
from sqlalchemy import create_engine, engine_from_config
from sqlalchemy import pool
from alembic import context

# Import SQLModel and your models
from sqlmodel import SQLModel
from db.models import (
    McpServerEntity,
    LlmModelEntity,
    EmbeddingModelEntity,
    WebSearchConfigEntity,
    TraceModelEntity,
    KbEntity,
    ThreadEntity,
    MessageEntity,
    KbFileEntity,
    KbChunkEntity,
    RerankerModelEntity,
    RoleEntity,
    UserRoleEntity,
    PermissionEntity,
    KbMetadataEntity,
    FileMetadataEntity,
    PromptModelEntity,
    ChatBotEntity,
    GuardrailConfigEntity,
    KbFileTaskEntity,
    DatasetEntity,
    DatasetSampleEntity,
    EvaluatorConfigEntity,
    ExperimentEntity,
    ExperimentSampleEntity,
    RunConfigEntity,
    VectorDbConfig,
    FileEntity,
    FileTextContentEntity,
    FileUploadSessionEntity,
    FileChunkEntity,
)


def get_sync_db_engine():
    # 从环境变量中读取数据库配置
    # 支持三种数据库类型：sqlite（默认）、postgresql、mysql
    # MySQL 必须使用 utf8mb4 编码，建议创建数据库时设置：
    # CREATE DATABASE your_database_name CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
    if not os.path.exists("./localdata"):
        os.makedirs("./localdata")
    db_type = os.getenv("DB_TYPE", "sqlite")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT")

    if db_type == "postgresql":
        assert db_name, "Postgres db_name不能为空。"
        assert db_host, "Postgres db_host不能为空。"
        assert db_user, "Postgres db_user不能为空。"
        assert db_password, "Postgres db_password不能为空。"
        db_port = int(db_port) if db_port else 5432

        encoded_db_user = quote_plus(db_user)
        encoded_db_password = quote_plus(db_password)

        db_url = f"postgresql+psycopg2://{encoded_db_user}:{encoded_db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(db_url, echo=False)
        logger.info(
            f"created sync engine with PostgreSQL {db_user}@{db_host}:{db_port}/{db_name}"
        )

        return engine
    elif db_type == "mysql":
        assert db_name, "MySQL db_name不能为空。"
        assert db_host, "MySQL db_host不能为空。"
        assert db_user, "MySQL db_user不能为空。"
        assert db_password, "MySQL db_password不能为空。"
        db_port = int(db_port) if db_port else 3306

        encoded_db_user = quote_plus(db_user)
        encoded_db_password = quote_plus(db_password)

        # MySQL 使用 pymysql 驱动，并设置 utf8mb4 编码
        db_url = f"mysql+pymysql://{encoded_db_user}:{encoded_db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
        engine = create_engine(
            db_url,
            echo=False,
            connect_args={
                "charset": "utf8mb4",
            },
        )
        logger.info(
            f"created sync engine with MySQL {db_user}@{db_host}:{db_port}/{db_name} (utf8mb4)"
        )

        return engine
    else:
        init_sqlite_store()
        local_db_url = os.getenv("SQLITE_URL", "sqlite:///./tmp/sqlite/local.db")

        logger.warning(
            f"Created db engine with sqlite {local_db_url}."
        )
        return create_engine(
            local_db_url,
            echo=False,  # 输出执行的 SQL 语句
            connect_args={"check_same_thread": False},  # SQLite 特有参数
        )


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


engine = get_sync_db_engine()
engine_url = engine.url.render_as_string(hide_password=False).replace("%", "%%")


config.set_main_option("sqlalchemy.url", engine_url)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# Set target_metadata to SQLModel's metadata
target_metadata = SQLModel.metadata
target_metadata.create_all(engine)
# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()