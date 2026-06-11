from db.models.llm import LlmModelEntity
from common.llm.llm_model import PaiLlm
from common.encrypt_utils import decrypt_key
from db.models.knowledgebase.embedding import EmbeddingModelEntity, EmbeddingType
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rag.embedding.multimodal_dashscope_embedding import MultimodalDashscopeEmbedding
from rag.rerank.dashscope_reranker import DashscopeReranker
from rag.rerank.multimodal_dashscope_reranker import MultimodalDashscopeReranker
from utils.modelscope_utils import download_model_to_directory
from rag.rerank.reranker import OpenAICompatibleReranker
from db.models.knowledgebase.reranker import RerankerType, RerankerModelEntity
from loguru import logger
from typing import Union
from utils.cuda_utils import infer_cuda_device
from common.llm.openai.openai_like import OpenAILike
from utils.lru_cache import LruCache


embedding_cache = LruCache(max_size=10)
llm_cache = LruCache(max_size=20)
reranker_cache = LruCache(max_size=10)


def llm_cache_key(config: LlmModelEntity) -> str:
    return f"llm_{config.base_url}_{config.encrypted_api_key}_{config.model}_{config.enable_thinking}_{config.vision_support}_{config.temperature}_{config.context_window}_{config.max_tokens}"

def create_llm(config: LlmModelEntity) -> PaiLlm:
    llm_key = llm_cache_key(config)
    llm = llm_cache.get(llm_key)
    if llm:
        logger.info(f"Using cached LLM model {llm_key}.")
        return llm

    logger.info(
        f"Creating PaiLlm model {config.model} with {config}."
    )
    llm = PaiLlm(
            api_base=config.base_url,
            api_key=decrypt_key(config.encrypted_api_key),
            model=config.model,
            enable_thinking=config.enable_thinking,
            vision_support=config.vision_support,
            temperature=config.temperature,
            context_window=config.context_window,
            max_tokens=config.max_tokens,
        )
    llm_cache.put(llm_key, llm)
    return llm


def create_openailike_llm(config: LlmModelEntity) -> OpenAILike:
    logger.info(
        f"Creating OpenAI like LLM model {config.model} with {config}."
    )
    return OpenAILike(
        model=config.model,
        api_base=config.base_url,
        api_key=decrypt_key(config.encrypted_api_key),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        is_chat_model=True,
        is_function_calling_model=True,
    )


def create_embedding_model(config: EmbeddingModelEntity) -> BaseEmbedding:
    def _embed_cache_key(config: EmbeddingModelEntity) -> str:
        if config.type == EmbeddingType.OPENAI_LIKE:
            return f"openailike_{config.endpoint}_{config.encrypted_api_key}_{config.model_name}_{config.dimension}_{config.embed_batch_size}"
        elif config.type == EmbeddingType.LOCAL:
            return f"local_{config.model_name}"
        elif config.type == EmbeddingType.MULTIMODAL_DASHSCOPE:
            return f"mm_dashscope_{config.endpoint}_{config.encrypted_api_key}_{config.model_name}_{config.dimension}_{config.embed_batch_size}"
        else:
            raise ValueError(f"Unsupported embedding type: {config.type}")

    embed_key = _embed_cache_key(config)
    embedding_model = embedding_cache.get(embed_key)
    if embedding_model:
        logger.info(f"Using cached embedding model {embed_key}.")
        return embedding_model

    if config.type == EmbeddingType.OPENAI_LIKE:
        logger.info(
            f"Creating OpenAI like embedding model {config.model_name} with {config}."
        )
        embedding_model = OpenAILikeEmbedding(
            api_key=decrypt_key(config.encrypted_api_key),
            model_name=config.model_name,
            dimensions=config.dimension,
            embed_batch_size=config.embed_batch_size,
            api_base=config.endpoint,
        )
    elif config.type == EmbeddingType.LOCAL:
        logger.info(
            f"Creating local embedding model {config.model_name} with {config}."
        )
        pai_model_path = download_model_to_directory(config.model_name)
        if not pai_model_path:
            raise ValueError(f"Failed to download model {config.model_name}.")
        embedding_model = HuggingFaceEmbedding(
            model_name=pai_model_path,
            embed_batch_size=config.embed_batch_size,
            device=infer_cuda_device(),
            show_progress_bar=False,
        )
    elif config.type == EmbeddingType.MULTIMODAL_DASHSCOPE:
        logger.info(
            f"Creating multimodal DashScope embedding model {config.model_name} with {config}."
        )
        embedding_model = MultimodalDashscopeEmbedding(
            api_key=decrypt_key(config.encrypted_api_key),
            model_name=config.model_name,
            base_url=config.endpoint or "",
            dimension=config.dimension,
            embed_batch_size=config.embed_batch_size,
        )
    else:
        raise ValueError(f"Unsupported embedding type: {config.type}")

    embedding_cache.put(embed_key, embedding_model)
    return embedding_model


def create_reranker_model(
    config: RerankerModelEntity,
) -> Union[DashscopeReranker, OpenAICompatibleReranker, MultimodalDashscopeReranker]:
    def reranker_cache_key(config: RerankerModelEntity) -> str:
        return f"reranker_{config.base_url}_{config.encrypted_api_key}_{config.model_name}_{config.type}"

    reranker_key = reranker_cache_key(config)
    reranker = reranker_cache.get(reranker_key)
    if reranker:
        logger.info(f"Using cached reranker model {reranker_key}.")
        return reranker

    model_type = config.type or RerankerType.OPENAI_LIKE

    if model_type == RerankerType.MULTIMODAL_DASHSCOPE:
        logger.info(
            f"Creating multimodal DashScope reranker model {config.model_name} with {config}."
        )
        reranker = MultimodalDashscopeReranker(
            api_key=decrypt_key(config.encrypted_api_key),
            model=config.model_name,
            base_url=config.base_url,
        )
    elif model_type == RerankerType.DASHSCOPE:
        logger.info(
            f"Creating DashScope reranker model {config.model_name} with {config}."
        )
        reranker = DashscopeReranker(
            api_key=decrypt_key(config.encrypted_api_key),
            model=config.model_name,
            base_url=config.base_url,
        )
    else:
        logger.info(
            f"Creating OpenAI compatible reranker model {config.model_name} with {config}."
        )
        reranker = OpenAICompatibleReranker(
            api_key=decrypt_key(config.encrypted_api_key),
            model=config.model_name,
            base_url=config.base_url,
        )

    reranker_cache.put(reranker_key, reranker)
    return reranker
