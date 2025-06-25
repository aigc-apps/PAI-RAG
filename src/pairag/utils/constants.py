"""Set of constants of modules."""

import os


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


EAS_DEFAULT_MODEL_DIR = "/huggingface/pai_rag_model_repository_01"
if not os.path.exists(EAS_DEFAULT_MODEL_DIR):
    DEFAULT_MODEL_DIR = "./localdata/model_repository"
else:
    DEFAULT_MODEL_DIR = EAS_DEFAULT_MODEL_DIR

OSS_URL = "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/model_repository/model_config_1.1.0.json"

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
