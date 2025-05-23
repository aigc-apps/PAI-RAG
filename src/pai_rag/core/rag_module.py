from typing import Any
from loguru import logger
from copy import deepcopy
import threading

from llama_index.core import Settings
from llama_index.core.query_engine import BaseQueryEngine

from pai_rag.core.rag_config import RagConfig
from pai_rag.extensions.news.miaobi_news import MiaobiNewsTool
from pai_rag.file.store.pai_image_store import PaiImageStore
from pai_rag.integrations.embeddings.pai.pai_embedding_config import (
    HuggingFaceEmbeddingConfig,
)
from pai_rag.data_pipeline.job.file_task_executor import FileTaskExecutor
from pai_rag.integrations.data_analysis.data_analysis_tool import (
    DataAnalysisConnector,
    DataAnalysisLoader,
    DataAnalysisQuery,
)
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding

# cnclip import should come before others. otherwise will segment fault.
from pai_rag.integrations.guardrail.pai_guardrail import PaiLlmGuardrail
from pai_rag.knowledgebase.index.pai.pai_vector_index import PaiVectorStoreIndex
from pai_rag.file.nodeparsers.pai.pai_node_parser import PaiNodeParser
from pai_rag.file.store.oss_store import PaiOssStore
from pai_rag.integrations.postprocessor.pai.pai_postprocessor import PaiPostProcessor
from pai_rag.integrations.query_engine.pai_retriever_query_engine import (
    PaiRetrieverQueryEngine,
)
from pai_rag.integrations.query_transform.pai_query_transform import (
    OpenAICompatibleQueryTransform,
)
from pai_rag.knowledgebase.index.pai.vector_store_config import FaissVectorStoreConfig
from pai_rag.knowledgebase.rag_knowledgebase import KnowledgeBase
from pai_rag.file.readers.pai.pai_data_reader import BaseDataReaderConfig, PaiDataReader
from pai_rag.integrations.search.bing_search import BingSearchTool
from pai_rag.integrations.search.quark_search import QuarkSearchTool
from pai_rag.integrations.search.aliyun_search import AliyunSearchTool
from pai_rag.integrations.search.google_search import GoogleSearchTool
from pai_rag.integrations.synthesizer.pai_synthesizer import PaiSynthesizer
from pai_rag.integrations.llms.pai.pai_llm import PaiLlm
from pai_rag.integrations.llms.pai.pai_multi_modal_llm import PaiMultiModalLlm
from pai_rag.file.nodeparsers.pai.image_caption_tool import ImageCaptionTool
from pai_rag.integrations.search.search_config import (
    BingSearchConfig,
    QuarkSearchConfig,
    AliyunSearchConfig,
    GoogleSearchConfig,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from pai_rag.integrations.postprocessor.pai.pai_postprocessor import PostProcessorType
from pai_rag.integrations.postprocessor.pai.pai_postprocessor import (
    SimilarityPostProcessorConfig,
    RerankModelPostProcessorConfig,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_RERANK_SIMILARITY_THRESHOLD,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_TOP_N,
)
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K

cls_cache = {}


def resolve_by_thread(cls: Any, thread: int, **kwargs):
    cls_kwargs = {"thread": thread}
    cls_kwargs.update(kwargs)
    cls_key = cls_kwargs.__repr__()
    if cls_key not in cls_cache:
        cls_cache[cls_key] = cls(**kwargs)
    return cls_cache[cls_key]


def resolve(cls: Any, **kwargs):
    cls_key = kwargs.__repr__()
    if cls_key not in cls_cache:
        cls_cache[cls_key] = cls(**kwargs)
    return cls_cache[cls_key]


def resolve_chat_llm(config: RagConfig, model_id: str = None) -> PaiLlm:
    if model_id == "default":
        model_id = None

    if model_id is not None:
        for llm_config in config.llms:
            if llm_config.is_validate() and (
                llm_config.model_id == model_id or llm_config.model == model_id
            ):
                if not llm_config.vision_support:
                    llm = resolve(cls=PaiLlm, llm_config=llm_config)
                else:
                    llm = resolve(cls=PaiMultiModalLlm, llm_config=llm_config)
                return llm

    if len(config.llms) > 0:
        llm_config = config.llms[0]
        if not llm_config.vision_support:
            llm = resolve(cls=PaiLlm, llm_config=llm_config)
        else:
            llm = resolve(cls=PaiMultiModalLlm, llm_config=llm_config)
        return llm
    else:
        logger.warning("No llm found")
        Settings.llm = None
        return None


def resolve_multimodal_llm(config: RagConfig, model_id: str = None) -> PaiMultiModalLlm:
    for vllm_config in config.llms:
        if vllm_config.is_validate() and vllm_config.vision_support:
            vllm = resolve(cls=PaiMultiModalLlm, llm_config=vllm_config)
            return vllm
    logger.info(f"No llm found for multimodal model_id: {model_id}")
    return None


def resolve_query_rewrite_llm(config: RagConfig, model_id: str = None) -> PaiLlm:
    model_id = model_id or config.query_rewrite.model_id
    if (
        config.query_rewrite.llm
        and config.query_rewrite.llm.base_url
        and config.query_rewrite.llm.api_key
        and config.query_rewrite.llm.model
    ):
        return resolve(cls=PaiLlm, llm_config=config.query_rewrite.llm)
    return resolve_chat_llm(config, model_id)


def resolve_news_extension_llm(config: RagConfig, model_id: str = None) -> PaiLlm:
    model_id = model_id or config.news_extension.model_id
    if (
        config.news_extension.llm
        and config.news_extension.llm.base_url
        and config.news_extension.llm.api_key
        and config.news_extension.llm.model
    ):
        return resolve(cls=PaiLlm, llm_config=config.news_extension.llm)
    return resolve_chat_llm(config, model_id)


def resolve_llm_guardrail(config: RagConfig) -> PaiLlmGuardrail:
    if config.guardrail.is_enabled():
        guardrail = resolve(
            cls=PaiLlmGuardrail,
            config=config.guardrail,
        )
        return guardrail

    return None


def resolve_default_embedding():
    return resolve(cls=PaiEmbedding, embed_config=HuggingFaceEmbeddingConfig())


def resolve_task_executor(
    config: RagConfig, knowledgebase: KnowledgeBase
) -> FileTaskExecutor:
    image_store = None
    if config.oss_store.bucket:
        oss_store = resolve(
            cls=PaiOssStore,
            bucket_name=config.oss_store.bucket,
            endpoint=config.oss_store.endpoint,
        )
        image_store = resolve(
            cls=PaiImageStore,
            oss_store=oss_store,
        )

    multimodal_llm = resolve_multimodal_llm(config)

    caption_tool = None
    if multimodal_llm:
        caption_tool = resolve(
            cls=ImageCaptionTool,
            multimodal_llm=multimodal_llm,
        )

    data_reader = resolve(
        cls=PaiDataReader,
        reader_config=BaseDataReaderConfig(),
        image_store=image_store,
    )

    node_parser = resolve(
        cls=PaiNodeParser,
        parser_config=knowledgebase.node_parser_config,
        caption_tool=caption_tool,
    )

    embed_model = resolve(cls=PaiEmbedding, embed_config=knowledgebase.embedding_config)

    if isinstance(knowledgebase.vector_store_config, FaissVectorStoreConfig):
        vector_index = resolve(
            cls=PaiVectorStoreIndex,
            vector_store_config=knowledgebase.vector_store_config,
            embed_model=embed_model,
        )
    else:
        vector_index = resolve_by_thread(
            cls=PaiVectorStoreIndex,
            vector_store_config=knowledgebase.vector_store_config,
            embed_model=embed_model,
            thread=threading.current_thread().ident,
        )

    logger.debug(
        f"create FileTaskExecutor with params [node_parser]: {node_parser}, [embed_model]: {embed_model}, [vector_index]: {vector_index}, [data_reader]: {data_reader}"
    )
    return resolve(
        cls=FileTaskExecutor,
        node_parser=node_parser,
        embed_model=embed_model,
        vector_index=vector_index,
        data_reader=data_reader,
    )


def resolve_data_analysis_connector(config: RagConfig):
    db_connector = resolve(
        cls=DataAnalysisConnector,
        analysis_config=config.data_analysis,
    )
    return db_connector


def resolve_data_analysis_loader(
    config: RagConfig, model_id: str = None
) -> DataAnalysisLoader:
    llm = resolve_chat_llm(config, model_id)
    sql_database = DataAnalysisConnector(
        config.data_analysis
    ).connect()  # 每次load都会重连数据库

    return resolve(
        cls=DataAnalysisLoader,
        analysis_config=config.data_analysis,
        sql_database=sql_database,
        embed_model=resolve_default_embedding(),
        llm=llm,
    )


def resolve_da_llm(config: RagConfig, model_id: str = None) -> PaiLlm:
    selected_llm_config = next(
        (config for config in config.llms if config.model_id == model_id), None
    )
    if selected_llm_config:
        llm_da_config = deepcopy(selected_llm_config)
        llm_da_config.max_tokens = 1024
        llm = resolve(cls=PaiLlm, llm_config=llm_da_config)
        return llm
    else:
        logger.warning("No llm found")
        Settings.llm = None
        return None


def resolve_data_analysis_query(
    config: RagConfig, model_id: str = None
) -> DataAnalysisQuery:
    llm_da = resolve_chat_llm(config, model_id)
    sql_database = resolve_data_analysis_connector(config).connect()

    return resolve(
        cls=DataAnalysisQuery,
        analysis_config=config.data_analysis,
        sql_database=sql_database,
        llm=llm_da,
        embed_model=resolve_default_embedding(),
        callback_manager=None,
    )


def resolve_openai_query_transform(
    config: RagConfig, model_id: str = None
) -> OpenAICompatibleQueryTransform:
    if not config.query_rewrite.enabled:
        return None
    llm = resolve_query_rewrite_llm(config, model_id)

    openai_query_transform = resolve(
        OpenAICompatibleQueryTransform,
        llm=llm,
        base_transform_prompt=config.query_rewrite.base_prompt_template_str,
        llm_tool_prompt_str=config.query_rewrite.llm_tool_prompt_str,
        knowledge_tool_prompt_str=config.query_rewrite.knowledge_tool_prompt_str,
        websearch_tool_prompt_str=config.query_rewrite.websearch_tool_prompt_str,
        agent_tool_prompt_str=config.query_rewrite.agent_tool_prompt_str,
        db_tool_prompt_str=config.query_rewrite.db_tool_prompt_str,
        news_tool_prompt_str=config.query_rewrite.news_tool_prompt_str.format(
            domain_list=config.news_extension.domain_list,
            news_role=config.news_extension.news_role,
        ),
        news_valid_domain_list=config.news_extension.domain_list,
    )
    return openai_query_transform


def resolve_synthesizer(config: RagConfig, model_id: str = None) -> PaiSynthesizer:
    llm = resolve_chat_llm(config, model_id)

    synthesizer = resolve(
        cls=PaiSynthesizer,
        llm=llm,
        system_role_template=config.synthesizer.system_role_template,
        custom_prompt_template=config.synthesizer.custom_prompt_template,
    )
    return synthesizer


def resolve_vector_index(knowledgebase: KnowledgeBase) -> PaiVectorStoreIndex:
    embed_model = resolve(cls=PaiEmbedding, embed_config=knowledgebase.embedding_config)
    if isinstance(knowledgebase.vector_store_config, FaissVectorStoreConfig):
        vector_index = resolve(
            cls=PaiVectorStoreIndex,
            vector_store_config=knowledgebase.vector_store_config,
            embed_model=embed_model,
        )
    else:
        vector_index = resolve_by_thread(
            cls=PaiVectorStoreIndex,
            vector_store_config=knowledgebase.vector_store_config,
            embed_model=embed_model,
            thread=threading.current_thread().ident,
        )

    return vector_index


def resolve_query_engine(
    config: RagConfig, vector_index: PaiVectorStoreIndex, model_id: str = None
) -> PaiRetrieverQueryEngine:
    retriever = vector_index.as_retriever()

    synthesizer = resolve_synthesizer(config, model_id)
    postprocessor = resolve(
        cls=PaiPostProcessor, postprocessor_config=config.postprocessor
    )

    query_engine = resolve(
        cls=PaiRetrieverQueryEngine,
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=[postprocessor],
        callback_manager=Settings.callback_manager,
    )

    return query_engine


def resolve_query_engine_from_retrieval_request(
    config: RagConfig,
    vector_index: PaiVectorStoreIndex,
    retrieval_settings: dict = {},
    model_id: str = None,
) -> PaiRetrieverQueryEngine:
    retrieval_mode = retrieval_settings.get(
        "retrieval_mode", VectorStoreQueryMode.DEFAULT
    )
    if isinstance(retrieval_mode, str):
        retrieval_mode = VectorStoreQueryMode(retrieval_mode)

    """
    hybrid_fusion_weights = [
        retrieval_settings.get(
            "vector_weight", 0.5
        ),
        retrieval_settings.get(
            "keyword_weight", 0.5
        ),
    ]
    """

    retriever = vector_index.as_retriever(
        vector_store_query_mode=retrieval_mode,
        similarity_top_k=retrieval_settings.get(
            "similarity_top_k", DEFAULT_SIMILARITY_TOP_K
        ),
    )

    synthesizer = resolve_synthesizer(config, model_id)

    _reranker_type = retrieval_settings.get(
        "reranker_type", config.postprocessor.reranker_type
    )

    if isinstance(_reranker_type, str):
        _reranker_type = PostProcessorType(_reranker_type)

    if _reranker_type == PostProcessorType.no_reranker:
        _postprocessor_config = SimilarityPostProcessorConfig(
            reranker_type=PostProcessorType.no_reranker,
            similarity_threshold=retrieval_settings.get(
                "similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD
            ),
        )
    elif _reranker_type == PostProcessorType.reranker_model:
        _postprocessor_config = RerankModelPostProcessorConfig(
            reranker_type=PostProcessorType.reranker_model,
            reranker_model=retrieval_settings.get(
                "reranker_model", DEFAULT_RERANK_MODEL
            ),
            top_n=retrieval_settings.get(
                "reranker_similarity_top_k", DEFAULT_RERANK_TOP_N
            ),
            similarity_threshold=retrieval_settings.get(
                "reranker_similarity_threshold", DEFAULT_RERANK_SIMILARITY_THRESHOLD
            ),
        )

    postprocessor = resolve(
        cls=PaiPostProcessor, postprocessor_config=_postprocessor_config
    )

    query_engine = resolve(
        cls=PaiRetrieverQueryEngine,
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=[postprocessor],
        callback_manager=Settings.callback_manager,
    )

    return query_engine


def resolve_query_engine_from_knowledgebase(
    config: RagConfig,
    vector_index: PaiVectorStoreIndex,
    knowledgebase: KnowledgeBase,
    model_id: str = None,
) -> PaiRetrieverQueryEngine:
    retrieval_settings = knowledgebase.retrieval_settings
    if retrieval_settings is not None:
        retrieval_mode = retrieval_settings.get(
            "retrieval_mode", VectorStoreQueryMode.DEFAULT
        )
        if isinstance(retrieval_mode, str):
            retrieval_mode = VectorStoreQueryMode(retrieval_mode)

        retriever = vector_index.as_retriever(
            vector_store_query_mode=retrieval_mode,
        )

        _reranker_type = retrieval_settings.get(
            "reranker_type", config.postprocessor.reranker_type
        )

        if isinstance(_reranker_type, str):
            _reranker_type = PostProcessorType(_reranker_type)
        if _reranker_type == PostProcessorType.no_reranker:
            _postprocessor_config = SimilarityPostProcessorConfig(
                reranker_type=PostProcessorType.no_reranker,
                similarity_threshold=retrieval_settings.get(
                    "similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD
                ),
            )
        elif _reranker_type == PostProcessorType.reranker_model:
            _postprocessor_config = RerankModelPostProcessorConfig(
                reranker_type=PostProcessorType.reranker_model,
                reranker_model=retrieval_settings.get(
                    "reranker_model", DEFAULT_RERANK_MODEL
                ),
                top_n=retrieval_settings.get(
                    "reranker_similarity_top_k", DEFAULT_RERANK_TOP_N
                ),
                similarity_threshold=retrieval_settings.get(
                    "reranker_similarity_threshold", DEFAULT_RERANK_SIMILARITY_THRESHOLD
                ),
            )

        postprocessor = resolve(
            cls=PaiPostProcessor, postprocessor_config=_postprocessor_config
        )
    else:
        retriever = vector_index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
        )
        postprocessor = resolve(
            cls=PaiPostProcessor, postprocessor_config=config.postprocessor
        )

    qa_prompt_templates = knowledgebase.qa_prompt_templates
    if qa_prompt_templates is not None:
        llm = resolve_chat_llm(config, model_id)

        synthesizer = resolve(
            cls=PaiSynthesizer,
            llm=llm,
            system_role_template=qa_prompt_templates["system_prompt_template"],
            custom_prompt_template=qa_prompt_templates["task_prompt_template"],
        )
    else:
        synthesizer = resolve_synthesizer(config, model_id)

    query_engine = resolve(
        cls=PaiRetrieverQueryEngine,
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=[postprocessor],
        callback_manager=Settings.callback_manager,
    )

    return query_engine


def resolve_searcher(config: RagConfig, model_id: str = None) -> BaseQueryEngine:
    synthesizer = resolve_synthesizer(config, model_id)
    searcher = None

    if isinstance(config.search, BingSearchConfig) and config.search.search_api_key:
        searcher = resolve(
            cls=BingSearchTool,
            api_key=config.search.search_api_key,
            synthesizer=synthesizer,
            search_count=config.search.search_count,
            search_lang=config.search.search_lang,
            search_qa_prompt_template=config.search.search_qa_prompt_template,
        )
    elif (
        isinstance(config.search, QuarkSearchConfig)
        and config.search.user
        and config.search.secret
    ):
        searcher = resolve(
            cls=QuarkSearchTool,
            user=config.search.user,
            secret=config.search.secret,
            host=config.search.host,
            synthesizer=synthesizer,
            search_count=config.search.search_count,
            search_qa_prompt_template=config.search.search_qa_prompt_template,
        )
    elif (
        isinstance(config.search, AliyunSearchConfig)
        and config.search.access_key_id
        and config.search.access_key_secret
    ):
        searcher = resolve(
            cls=AliyunSearchTool,
            access_key_id=config.search.access_key_id,
            access_key_secret=config.search.access_key_secret,
            endpoint=config.search.endpoint,
            synthesizer=synthesizer,
            search_count=config.search.search_count,
            search_qa_prompt_template=config.search.search_qa_prompt_template,
        )
    elif isinstance(config.search, GoogleSearchConfig) and config.search.serpapi_key:
        searcher = resolve(
            cls=GoogleSearchTool,
            api_key=config.search.serpapi_key,
            synthesizer=synthesizer,
            search_count=config.search.search_count,
            search_lang=config.search.search_lang,
            search_qa_prompt_template=config.search.search_qa_prompt_template,
        )

    return searcher


def resolve_news_tool(config: RagConfig, model_id: str = None) -> MiaobiNewsTool:
    if config.news_extension.is_enabled():
        llm = resolve_news_extension_llm(config, model_id)
        news_tool = resolve(
            cls=MiaobiNewsTool,
            llm=llm,
            config=config.news_extension,
        )
        return news_tool

    return None
