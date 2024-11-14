from typing import Any

from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from pai_rag.core.models.config import ArizeTraceConfig, BaseTraceConfig, PaiTraceConfig
from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_data_loader import RagDataLoader
from pai_rag.integrations.agent.pai.pai_agent import PaiAgent
from pai_rag.integrations.chat_store.pai.pai_chat_store import PaiChatStore
from pai_rag.integrations.data_analysis.data_analysis_tool import DataAnalysisTool
from pai_rag.integrations.data_analysis.data_analysis_tool1 import (
    DataAnalysisConnector,
    DataAnalysisLoader,
    DataAnalysisQuery,
)
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding

# cnclip import should come before others. otherwise will segment fault.
from pai_rag.integrations.embeddings.pai.pai_multimodal_embedding import (
    PaiMultiModalEmbedding,
)
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import PaiNodeParser
from pai_rag.integrations.nodes.raptor_nodes_enhance import RaptorProcessor
from pai_rag.integrations.postprocessor.pai.pai_postprocessor import PaiPostProcessor
from pai_rag.integrations.query_engine.pai_retriever_query_engine import (
    PaiRetrieverQueryEngine,
)
from pai_rag.integrations.query_transform.pai_query_transform import (
    PaiCondenseQueryTransform,
)
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
from pai_rag.integrations.router.pai.pai_router import PaiIntentRouter
from pai_rag.integrations.search.bing_search import BingSearchTool
from pai_rag.integrations.synthesizer.pai_synthesizer import PaiSynthesizer
from pai_rag.integrations.llms.pai.pai_llm import PaiLlm
from pai_rag.integrations.llms.pai.pai_multi_modal_llm import PaiMultiModalLlm
from pai_rag.utils.oss_client import OssClient
import logging

logger = logging.getLogger(__name__)

cls_cache = {}


def resolve(cls: Any, **kwargs):
    cls_key = kwargs.__repr__()
    if cls_key not in cls_cache:
        cls_cache[cls_key] = cls(**kwargs)
    return cls_cache[cls_key]


def resolve_chat_store(config: RagConfig) -> PaiChatStore:
    chat_store = resolve(PaiChatStore, chat_store_config=config.chat_store)
    return chat_store


def resolve_intent_router(config: RagConfig) -> PaiIntentRouter:
    llm = resolve(cls=PaiLlm, llm_config=config.llm)
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

    data_reader = resolve(
        cls=PaiDataReader,
        reader_config=config.data_reader,
        oss_store=oss_store,
    )

    node_parser = resolve(cls=PaiNodeParser, parser_config=config.node_parser)

    embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)
    multimodal_embed_model = None
    if config.index.enable_multimodal:
        multimodal_embed_model = resolve(
            cls=PaiMultiModalEmbedding,
            multimodal_embed_config=config.multimodal_embedding,
        )

    vector_index = resolve(
        cls=PaiVectorStoreIndex,
        vector_store_config=config.index.vector_store,
        enable_multimodal=config.index.enable_multimodal,
        embed_model=embed_model,
        multimodal_embed_model=multimodal_embed_model,
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
        multimodal_embed_model=multimodal_embed_model,
        vector_index=vector_index,
    )

    return data_loader


def resolve_agent(config: RagConfig) -> PaiAgent:
    llm = resolve(cls=PaiLlm, llm_config=config.llm)
    agent = PaiAgent.from_tools(
        agent_config=config.agent,
        llm=llm,
    )
    return agent


def resolve_llm(config: RagConfig) -> PaiLlm:
    llm = resolve(cls=PaiLlm, llm_config=config.llm)
    Settings.llm = llm
    return llm


def resolve_data_analysis_tool(config: RagConfig) -> DataAnalysisTool:
    llm = resolve_llm(config)
    embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)

    return resolve(
        cls=DataAnalysisTool,
        analysis_config=config.data_analysis,
        llm=llm,
        embed_model=embed_model,
    )


def resolve_data_analysis_connector(config: RagConfig):
    db_connector = resolve(
        cls=DataAnalysisConnector,
        analysis_config=config.data_analysis,
    )
    # db_connector = DataAnalysisConnector(config.data_analysis)
    return db_connector


def resolve_data_analysis_loader(config: RagConfig) -> DataAnalysisLoader:
    llm = resolve_llm(config)
    embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)
    sql_database = resolve_data_analysis_connector(config).connect_db()
    # print("loader sql_database:", sql_database)

    return resolve(
        cls=DataAnalysisLoader,
        analysis_config=config.data_analysis,
        sql_database=sql_database,
        llm=llm,
        embed_model=embed_model,
    )


def resolve_data_analysis_query(config: RagConfig) -> DataAnalysisQuery:
    llm = resolve_llm(config)
    embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)
    sql_database = resolve_data_analysis_connector(config).connect_db()
    # print("query sql_database:", sql_database)

    return resolve(
        cls=DataAnalysisQuery,
        analysis_config=config.data_analysis,
        sql_database=sql_database,
        llm=llm,
        embed_model=embed_model,
        callback_manager=None,
    )


def resolve_query_transform(config: RagConfig) -> PaiCondenseQueryTransform:
    chat_store = resolve(PaiChatStore, chat_store_config=config.chat_store)
    llm = resolve_llm(config)
    condense_query_transform = resolve(
        PaiCondenseQueryTransform, llm=llm, chat_store=chat_store
    )
    return condense_query_transform


def resolve_synthesizer(config: RagConfig) -> PaiSynthesizer:
    llm = resolve(cls=PaiLlm, llm_config=config.llm)
    Settings.llm = llm
    multimodal_llm = None
    if config.multimodal_llm and config.synthesizer.use_multimodal_llm:
        multimodal_llm = resolve(cls=PaiMultiModalLlm, llm_config=config.multimodal_llm)
    synthesizer = resolve(
        cls=PaiSynthesizer,
        llm=llm,
        multimodal_llm=multimodal_llm,
        text_qa_template=PromptTemplate(template=config.synthesizer.text_qa_template),
        multimodal_qa_template=PromptTemplate(
            template=config.synthesizer.multimodal_qa_template
        ),
    )
    return synthesizer


def resolve_vector_index(config: RagConfig) -> PaiVectorStoreIndex:
    embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)
    multimodal_embed_model = None
    if config.index.enable_multimodal:
        multimodal_embed_model = resolve(
            cls=PaiMultiModalEmbedding,
            multimodal_embed_config=config.multimodal_embedding,
        )

    vector_index = resolve(
        cls=PaiVectorStoreIndex,
        vector_store_config=config.index.vector_store,
        enable_multimodal=config.index.enable_multimodal,
        embed_model=embed_model,
        multimodal_embed_model=multimodal_embed_model,
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


def resolve_searcher(config: RagConfig) -> BingSearchTool:
    embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)
    synthesizer = resolve_synthesizer(config)

    searcher = resolve(
        cls=BingSearchTool,
        api_key=config.search.search_api_key,
        synthesizer=synthesizer,
        embed_model=embed_model,
        search_count=config.search.search_count,
        search_lang=config.search.search_lang,
    )

    return searcher


def setup_tracing(trace_config: BaseTraceConfig):
    from pai.llm_trace.instrumentation import init_opentelemetry
    from pai.llm_trace.instrumentation.llama_index import LlamaIndexInstrumentor
    from llama_index.core import set_global_handler

    if isinstance(trace_config, PaiTraceConfig):
        if not trace_config.token:
            logger.info("Trace is not enabled since PaiTraceConfig.token is not set.")
            return

        init_opentelemetry(
            LlamaIndexInstrumentor,
            grpc_endpoint=trace_config.endpoint,
            token=trace_config.token,
            service_name=trace_config.app_name,
            service_version="0.1.0",
            service_id="",
            deployment_environment="",
            service_owner_id="",
            service_owner_sub_id="",
        )
        logger.info(f"Pai-LLM-Trace enabled with endpoint: '{trace_config.endpoint}'.")
    elif isinstance(trace_config, ArizeTraceConfig):
        set_global_handler("arize_phoenix")
        logger.info("Arize trace enabled.")
    else:
        logger.warning("Trace is not configured.")
