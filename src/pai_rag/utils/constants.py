"""Set of constants of modules."""

import os

# paragraph separator for splitter
DEFAULT_NODE_PARSER_TYPE = "Token"
DEFAULT_PARAGRAPH_SEP = "\n\n\n"
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

DEFAULT_MAX_INDEX_ENTRY_COUNT = os.environ.get("DEFAULT_MAX_INDEX_ENTRY_COUNT", 20)

DEFAULT_KNOWLEDGEBASE_PATH = "localdata/knowledgebase"
DEFAULT_KNOWLEDGEBASE_NAME = "default"
DEFAULT_KNOWLEDGEBASE_NAME_OLD = "default_index"
DEFAULT_KNOWLEDGEBASE_FILE = "localdata/default__rag__index.json"
DEFAULT_DOC_STORE_NAME = "default__knowledge__docs.json"
DEFAULT_MAX_KNOWLEDGEBASE_COUNT = os.environ.get("DEFAULT_MAX_KNOWLEDGEBASE_COUNT", 20)
DEFAILT_MAX_FILE_TASK_COUNT = os.environ.get("DEFAILT_MAX_FILE_TASK_COUNT", 10000)
