"""Set of constants of modules."""

# paragraph separator for splitter
DEFAULT_PARAGRAPH_SEP = "\n\n\n"
SENTENCE_CHUNK_OVERLAP = 200
DEFAULT_WINDOW_SIZE = 3
DEFAULT_BREAKPOINT = 95
DEFAULT_BUFFER_SIZE = 1

DEFAULT_MODEL_DIR = "./huggingface/models"


DEFAULT_MODEL_DIC = {
    "SGPT-125M-weightedmean-nli-bitfit": "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/huggingface/compressed_models/SGPT-125M-weightedmean-nli-bitfit.zip",
    "bge-large-zh-v1.5": "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/huggingface/compressed_models/bge-large-zh-v1.5.zip",
    "bge-m3": "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/huggingface/compressed_models/bge-m3.zip",
    "bge-reranker-base": "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/huggingface/compressed_models/bge-reranker-base.zip",
    "bge-reranker-large": "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/huggingface/compressed_models/bge-reranker-large.zip",
    "bge-small-zh-v1.5": "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/huggingface/compressed_models/bge-small-zh-v1.5.zip",
    "paraphrase-multilingual-MiniLM-L12-v2": "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/huggingface/compressed_models/paraphrase-multilingual-MiniLM-L12-v2.zip",
    "qwen_1.8b": "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/huggingface/compressed_models/qwen_1.8b.zip",
    "text2vec-base-chinese": "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/huggingface/compressed_models/text2vec-base-chinese.zip",
    "text2vec-large-chinese": "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/huggingface/compressed_models/text2vec-large-chinese.zip",
    "easyocr": "https://pai-rag-bj.oss-cn-beijing.aliyuncs.com/huggingface/compressed_models/easyocr.zip",
}
