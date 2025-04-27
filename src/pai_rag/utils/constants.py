"""Set of constants of modules."""

import os
from loguru import logger


def try_get_int_env(key, default_value=None):
    """
    Retrieves an integer from an environment variable.
    """

    value_str = os.getenv(key)
    if value_str is None:
        if default_value is not None:
            return default_value
        else:
            return None
    try:
        return int(value_str)
    except ValueError:
        return None


# paragraph separator for splitter
DEFAULT_NODE_PARSER_TYPE = "Token"
DEFAULT_PARAGRAPH_SEP = "\n\n"
DEFAULT_SENTENCE_CHUNK_OVERLAP = 200
DEFAULT_SENTENCE_WINDOW_SIZE = 3
DEFAULT_BREAKPOINT = 95
DEFAULT_BUFFER_SIZE = 1

EAS_DEFAULT_MODEL_DIR = "/huggingface/pai_rag_model_repository_01"
if not os.path.exists(EAS_DEFAULT_MODEL_DIR):
    DEFAULT_MODEL_DIR = "./model_repository"
else:
    DEFAULT_MODEL_DIR = EAS_DEFAULT_MODEL_DIR

OSS_URL = (
    "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/model_repository/model_config.json"
)

DEFAULT_DATAFILE_DIR = "./data"

DEFAULT_DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v2"


DEFAULT_TASK_FILE = "localdata/ingestion__task__summary.json"

DEFAULT_INDEX_FILE = "localdata/default__rag__index.json"
DEFAULT_INDEX_NAME = "default"
DEFAULT_INDEX_NAME_OLD = "default_index"

DEFAULT_KNOWLEDGEBASE_PATH = "localdata/knowledgebase"
DEFAULT_KNOWLEDGEBASE_NAME = "default"
DEFAULT_KNOWLEDGEBASE_NAME_OLD = "default_index"
DEFAULT_KNOWLEDGEBASE_FILE = "localdata/default__rag__index.json"
DEFAULT_DOC_STORE_NAME = "default__knowledge__docs.json"
DEFAULT_MAX_KNOWLEDGEBASE_COUNT = try_get_int_env(
    "DEFAULT_MAX_KNOWLEDGEBASE_COUNT", 3000
)
DEFAULT_MAX_FILE_TASK_COUNT = try_get_int_env("DEFAULT_MAX_FILE_TASK_COUNT", 10000)

logger.info(
    f"""knowledgebase constants: DEFAULT_MAX_KNOWLEDGEBASE_COUNT: {DEFAULT_MAX_KNOWLEDGEBASE_COUNT} DEFAULT_MAX_FILE_TASK_COUNT: {DEFAULT_MAX_FILE_TASK_COUNT}
"""
)
