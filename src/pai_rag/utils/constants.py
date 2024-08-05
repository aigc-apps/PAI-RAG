"""Set of constants of modules."""

import os

# paragraph separator for splitter
DEFAULT_PARAGRAPH_SEP = "\n\n\n"
SENTENCE_CHUNK_OVERLAP = 200
DEFAULT_WINDOW_SIZE = 3
DEFAULT_BREAKPOINT = 95
DEFAULT_BUFFER_SIZE = 1

EAS_DEFAULT_MODEL_DIR = "/huggingface/pai_rag_model_repository"
if not os.path.exists(EAS_DEFAULT_MODEL_DIR):
    DEFAULT_MODEL_DIR = "./model_repository"
else:
    DEFAULT_MODEL_DIR = EAS_DEFAULT_MODEL_DIR

OSS_URL = (
    "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/model_repository/model_config.json"
)

DEFAULT_DATAFILE_DIR = "./data"
