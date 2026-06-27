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


DEFAULT_MODEL_DIR = "./localdata/model_repository"


DEFAULT_KNOWLEDGEBASE_PATH = "localdata/knowledgebase"
DEFAULT_KNOWLEDGEBASE_NAME = "default"
DEFAULT_KNOWLEDGEBASE_NAME_OLD = "default_index"
DEFAULT_KNOWLEDGEBASE_FILE = "localdata/default__rag__index.json"
DEFAULT_DOC_STORE_NAME = "default__knowledge__docs.json"
DEFAULT_MAX_KNOWLEDGEBASE_COUNT = try_get_int_env(
    "DEFAULT_MAX_KNOWLEDGEBASE_COUNT", 3000
)
DEFAULT_MAX_FILE_TASK_COUNT = try_get_int_env("DEFAULT_MAX_FILE_TASK_COUNT", 10000)
