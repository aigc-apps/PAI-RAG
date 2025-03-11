from typing import Any, List

from llama_index.core import Settings
from llama_index.core.query_engine import BaseQueryEngine

from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_data_loader import RagDataLoader
from pai_rag.integrations.agent.pai.pai_agent import PaiAgent
from pai_rag.integrations.chat_store.pai.pai_chat_store import PaiChatStore
from pai_rag.integrations.data_analysis.data_analysis_tool import (
    DataAnalysisConnector,
    DataAnalysisLoader,
    DataAnalysisQuery,
)
from pai_rag.integrations.llms.pai.llm_config import parse_llm_config
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding

# cnclip import should come before others. otherwise will segment fault.
from pai_rag.integrations.guardrail.pai_guardrail import PaiLlmGuardrail
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import PaiNodeParser
from pai_rag.integrations.nodes.raptor_nodes_enhance import RaptorProcessor
from pai_rag.integrations.postprocessor.pai.pai_postprocessor import PaiPostProcessor
from pai_rag.integrations.query_engine.pai_retriever_query_engine import (
    PaiRetrieverQueryEngine,
)
from pai_rag.integrations.query_transform.pai_query_transform import (
    OpenAICompatibleQueryTransform,
)
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
from pai_rag.integrations.router.pai.pai_router import (
    PaiIntentRouter,
)
from pai_rag.integrations.search.bing_search import BingSearchTool
from pai_rag.integrations.search.quark_search import QuarkSearchTool
from pai_rag.integrations.search.aliyun_search import AliyunSearchTool
from pai_rag.integrations.search.google_search import GoogleSearchTool
from pai_rag.integrations.synthesizer.pai_synthesizer import PaiSynthesizer
from pai_rag.integrations.llms.pai.pai_llm import PaiLlm
from pai_rag.integrations.llms.pai.pai_multi_modal_llm import PaiMultiModalLlm
from pai_rag.utils.oss_client import OssClient
from pai_rag.utils.image_caption_utils import ImageCaptionTool
from pai_rag.integrations.search.search_config import (
    BingSearchConfig,
    QuarkSearchConfig,
    AliyunSearchConfig,
    GoogleSearchConfig,
)


cls_cache = {}


def resolve(cls: Any, **kwargs):
    cls_key = kwargs.__repr__()
    if cls_key not in cls_cache:
        cls_cache[cls_key] = cls(**kwargs)
    return cls_cache[cls_key]


def resolve_llms(config: RagConfig) -> List[PaiLlm]:
    llms = [resolve(cls=PaiLlm, llm_config=llm_config) for llm_config in config.llms]
    return llms


def resolve_chat_llm(config: RagConfig) -> PaiLlm:
    # 兼容之前的配置
    if config.llm and all(
        [
            config.llm.source not in [None, ""],
            config.llm.api_key not in [None, ""],
            config.llm.base_url not in [None, ""],
            config.llm.model not in [None, ""],
        ]
    ):
        llm = resolve(cls=PaiLlm, llm_config=config.llm)
        Settings.llm = llm
        return llm
    # 新配置
    llms = resolve_llms(config)
    for llm in llms:
        if llm.model == config.chat.model_id:
            Settings.llm = llm
            return llm
    Settings.llm = llm[0]
    return llms[0]


def resolve_multimodal_llms(config: RagConfig) -> List[PaiMultiModalLlm]:
    # 兼容之前的配置
    if config.multimodal_llm and all(
        [
            config.multimodal_llm.source not in [None, ""],
            config.multimodal_llm.api_key not in [None, ""],
            config.multimodal_llm.base_url not in [None, ""],
            config.multimodal_llm.model not in [None, ""],
        ]
    ):
        multimodal_llm = resolve(cls=PaiMultiModalLlm, llm_config=config.multimodal_llm)
        return [multimodal_llm]
    # 新配置
    multimodal_llms = [
        resolve(cls=PaiMultiModalLlm, llm_config=llm_config)
        for llm_config in config.llms
        if llm_config.vision_support
    ]
    return multimodal_llms


def resolve_query_rewrite_llm(config: RagConfig):
    llms = resolve_llms(config)
    for llm in llms:
        if llm.model == config.query_rewrite.model_id:
            return llm
    return resolve_chat_llm(config)


# def resolve_llm(config: RagConfig) -> PaiLlm:
#     llm = resolve(cls=PaiLlm, llm_config=config.llm)
#     Settings.llm = llm
#     return llm


def resolve_llm_guardrail(config: RagConfig) -> PaiLlmGuardrail:
    if config.guardrail.is_enabled():
        guardrail = resolve(
            cls=PaiLlmGuardrail,
            config=config.guardrail,
        )
        return guardrail

    return None


def resolve_chat_store(config: RagConfig) -> PaiChatStore:
    chat_store = resolve(PaiChatStore, chat_store_config=config.chat_store)
    return chat_store


def resolve_intent_router(config: RagConfig) -> PaiIntentRouter:
    llm = resolve_chat_llm(config)
    intent_router = resolve(cls=PaiIntentRouter, intent_config=config.intent, llm=llm)
    return intent_router


def resolve_data_loader(config: RagConfig) -> RagDataLoader:
    oss_store = None
    if config.oss_store.bucket:
        oss_store = resolve(
            cls=OssClient,
            bucket_name=config.oss_store.bucket,
            endpoint=config.oss_store.endpoint,
        )

    multimodal_llms = resolve_multimodal_llms(config)

    caption_tool = None
    if multimodal_llms is not None and len(multimodal_llms) > 0:
        caption_tool = resolve(
            cls=ImageCaptionTool,
            multimodal_llm=multimodal_llms[0],
        )

    data_reader = resolve(
        cls=PaiDataReader,
        reader_config=config.data_reader,
        oss_store=oss_store,
    )

    node_parser = resolve(
        cls=PaiNodeParser, parser_config=config.node_parser, caption_tool=caption_tool
    )

    embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)

    vector_index = resolve(
        cls=PaiVectorStoreIndex,
        vector_store_config=config.index.vector_store,
        embed_model=embed_model,
        enable_local_keyword_index=True,
    )

    raptor_processor = resolve(
        cls=RaptorProcessor,
        tree_depth=config.node_enhancement.tree_depth,
        max_clusters=config.node_enhancement.max_clusters,
        threshold=config.node_enhancement.proba_threshold,
        embed_model=embed_model,
    )

    data_loader = RagDataLoader(
        data_reader=data_reader,
        node_parser=node_parser,
        raptor_processor=raptor_processor,
        embed_model=embed_model,
        vector_index=vector_index,
    )

    return data_loader


def resolve_agent(config: RagConfig) -> PaiAgent:
    llm = resolve(cls=PaiLlm, llm_config=config.llm)
    agent = resolve(
        cls=PaiAgent.from_tools,
        agent_config=config.agent,
        llm=llm,
    )
    return agent


def resolve_data_analysis_connector(config: RagConfig):
    db_connector = resolve(
        cls=DataAnalysisConnector,
        analysis_config=config.data_analysis,
    )
    return db_connector


def resolve_data_analysis_loader(config: RagConfig) -> DataAnalysisLoader:
    llm = resolve_chat_llm(config)
    sql_database = DataAnalysisConnector(
        config.data_analysis
    ).connect()  # 每次load都会重连数据库

    return resolve(
        cls=DataAnalysisLoader,
        analysis_config=config.data_analysis,
        sql_database=sql_database,
        llm=llm,
    )


def resolve_data_analysis_query(config: RagConfig) -> DataAnalysisQuery:
    if (
        config.data_analysis.llm
        and config.data_analysis.llm.base_url
        and config.data_analysis.llm.api_key
        and config.data_analysis.llm.model
    ):
        llm_da = resolve(cls=PaiLlm, llm_config=config.data_analysis.llm)
    else:
        llm_da_config = {
            "source": config.llm.source,
            "model": config.llm.model,
            "api_key": config.llm.api_key,
            "max_tokens": 1024,
        }
        llm_da = resolve(cls=PaiLlm, llm_config=parse_llm_config(llm_da_config))

    sql_database = resolve_data_analysis_connector(config).connect()

    return resolve(
        cls=DataAnalysisQuery,
        analysis_config=config.data_analysis,
        sql_database=sql_database,
        llm=llm_da,
        callback_manager=None,
    )


def resolve_openai_query_transform(config: RagConfig) -> OpenAICompatibleQueryTransform:
    if not config.query_rewrite.enabled:
        return None
    if config.query_rewrite.llm and all(
        [
            config.query_rewrite.llm.base_url not in [None, ""],
            config.query_rewrite.llm.api_key not in [None, ""],
            config.query_rewrite.llm.model not in [None, ""],
        ]
    ):
        llm = resolve(cls=PaiLlm, llm_config=config.query_rewrite.llm)
    elif config.llm and all(
        [
            config.llm.source not in [None, ""],
            config.llm.api_key not in [None, ""],
            config.llm.base_url not in [None, ""],
            config.llm.model not in [None, ""],
        ]
    ):
        llm = resolve(cls=PaiLlm, llm_config=config.llm)
    else:
        llm = resolve_query_rewrite_llm(config)

    openai_query_transform = resolve(
        OpenAICompatibleQueryTransform,
        llm=llm,
        query_transform_prompt=config.query_rewrite.rewrite_prompt_template,
    )
    return openai_query_transform


def resolve_synthesizer(config: RagConfig) -> PaiSynthesizer:
    llm = resolve_chat_llm(config)
    Settings.llm = llm
    multimodal_llm = None
    multimodal_llms = resolve_multimodal_llms(config)
    if (
        multimodal_llms
        and len(multimodal_llms) > 0
        and config.synthesizer.use_multimodal_llm
    ):
        multimodal_llm = multimodal_llms[0]

    synthesizer = resolve(
        cls=PaiSynthesizer,
        llm=llm,
        multimodal_llm=multimodal_llm,
        system_role_template=config.synthesizer.system_role_template,
        custom_prompt_template=config.synthesizer.custom_prompt_template,
    )
    return synthesizer


def resolve_vector_index(config: RagConfig) -> PaiVectorStoreIndex:
    embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)
    vector_index = resolve(
        cls=PaiVectorStoreIndex,
        vector_store_config=config.index.vector_store,
        embed_model=embed_model,
        enable_local_keyword_index=True,
    )
    return vector_index


def resolve_query_engine(config: RagConfig) -> PaiRetrieverQueryEngine:
    vector_index = resolve_vector_index(config)

    retriever = vector_index.as_retriever(
        vector_store_query_mode=config.retriever.vector_store_query_mode,
        similarity_top_k=config.retriever.similarity_top_k,
        image_similarity_top_k=config.retriever.image_similarity_top_k,
        search_image=config.retriever.search_image,
        hybrid_fusion_weights=config.retriever.hybrid_fusion_weights,
    )

    synthesizer = resolve_synthesizer(config)
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


def resolve_searcher(config: RagConfig) -> BaseQueryEngine:
    synthesizer = resolve_synthesizer(config)
    searcher = None

    if isinstance(config.search, BingSearchConfig) and config.search.search_api_key:
        searcher = resolve(
            cls=BingSearchTool,
            api_key=config.search.search_api_key,
            synthesizer=synthesizer,
            search_count=config.search.search_count,
            search_lang=config.search.search_lang,
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
        )
    elif isinstance(config.search, GoogleSearchConfig) and config.search.serpapi_key:
        searcher = resolve(
            cls=GoogleSearchTool,
            api_key=config.search.serpapi_key,
            synthesizer=synthesizer,
            search_count=config.search.search_count,
            search_lang=config.search.search_lang,
        )

    return searcher
