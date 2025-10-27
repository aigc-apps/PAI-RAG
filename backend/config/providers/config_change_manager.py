

from datetime import datetime, timezone
import asyncio
from typing import List
from sqlmodel import select
import traceback
from loguru import logger
from tenacity import retry, stop_after_attempt
from db.db_context import with_async_db_session
from db.models.change_event import ChangeEvent, ChangeEventSource, ChangeEventType
from config.providers.base_provider import BaseConfigProvider
from sqlmodel.ext.asyncio.session import AsyncSession
from config.providers.embedding_provider import embedding_provider
from config.providers.llm_provider import llm_provider
from config.providers.evaluation_provider import evaluation_provider
from db.models.knowledgebase.embedding import (
    EmbeddingModelCreate,
    EmbeddingModelEntity,
    EmbeddingType,
)
from db.models.knowledgebase.knowledgebase import (
    ChunkConfig,
    KbEntity,
    KnowledgebaseCreate,
    RetrievalConfig,
)
from db.models.evaluation.dataset import DatasetCreate, DatasetEntity
from db.models.evaluation.dataset import DatasetSampleEntity
from config.providers.knowledgebase_provider import knowledgebase_provider
from sqlalchemy.exc import IntegrityError
from rag.evaluation_tool import eval_client


class ConfigChangeManager:
    def __init__(self, worker_mode: bool = False):
        self.last_change_dt = None # 表示状态未初始化，不会扫描ChangeEvent表
        self.initialized = False
        self.worker_mode = worker_mode # 只需要管理embedding/llm/kb

    async def init_configuration(self):
        if self.initialized:
            return

        current_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        from db.db_context import init_db

        await init_db()
        logger.info("Initialized database tables.")

        await llm_provider.full_load_from_db_async()
        logger.info("Initialized llm models.")
        await self.create_default_embedding_model()
        await embedding_provider.full_load_from_db_async()
        logger.info("Initialized embedding models.")
        await knowledgebase_provider.full_load_from_db_async()
        logger.info("Initialized knowledgebases.")

        if not self.worker_mode:
            from config.providers.mcp_tool_provider import mcp_provider
            from config.providers.websearch_provider import websearch_provider
            from config.providers.trace_provider import trace_provider
            from config.providers.reranker_provider import reranker_provider
            from config.providers.chatbot_provider import chatbot_provider
            from config.providers.guardrail_provider import guardrail_provider
            from config.providers.vectordb_provider import vectordb_provider
            from config.providers.code_sandbox_provider import codesandbox_provider
            from config.providers.chatdb_provider import chatdb_provider

            await mcp_provider.full_load_from_db_async()
            logger.info("Initialized mcp tools.")
            await websearch_provider.full_load_from_db_async()
            logger.info("Initialized websearch configs.")
            await reranker_provider.full_load_from_db_async()
            logger.info("Initialized reranker configs.")
            await chatbot_provider.full_load_from_db_async()
            logger.info("Initialized chatbot configs.")
            await trace_provider.full_load_from_db_async()
            logger.info("Initialized trace configs.")
            await guardrail_provider.full_load_from_db_async()
            logger.info("Initialized guardrail configs.")
            await vectordb_provider.full_load_from_db_async()
            logger.info("Initialized vector db configs.")
            await codesandbox_provider.full_load_from_db_async()
            logger.info("Initialized code sandbox configs.")
            await chatdb_provider.full_load_from_db_async()
            logger.info("Initialized chatdb configs.")

        await llm_provider.full_load_from_db_async()
        logger.info("Initialized llm models.")
        await self.create_default_embedding_model()
        await embedding_provider.full_load_from_db_async()
        logger.info("Initialized embedding models.")
        await self.create_default_chat_doc_kb()
        await knowledgebase_provider.full_load_from_db_async()
        logger.info("Initialized knowledgebases.")



        await self.create_builtin_gaia_dataset()
        await evaluation_provider.full_load_from_db_async()
        logger.info("Initialized evaluation tasks.")

        self.initialized = True
        self.last_change_dt = current_dt
        logger.info(f"ConfigManager inited with worker_mode {self.worker_mode}, timestamp {self.last_change_dt}")


    @with_async_db_session
    async def create_default_chat_doc_kb(self, session: AsyncSession):
        # create default chat doc kb if not exists
        sql_results = await session.execute(
        select(KbEntity).where(KbEntity.name == "default_chat_docs")
        )
        chat_doc_entities: List[KbEntity] = sql_results.all()
        if len(chat_doc_entities) > 0:
            logger.info("Default chat doc knowledgebase already exists.")
            return

        logger.info("Creating default chat doc knowledgebase.")
        kb = KnowledgebaseCreate(
            name="default_chat_docs",
            description="聊天中生成的文档",
            embedding_model="BAAI/bge-m3",
        )
        kb.chunk_config = (ChunkConfig()).model_dump()
        kb.retrieval_config = (RetrievalConfig()).model_dump()
        knowledgebase = KbEntity.model_validate(kb)
        try:
            knowledgebase_provider.add(knowledgebase)
            session.add(knowledgebase)
            await session.commit()
            await session.refresh(knowledgebase)
            await self.notify_change_async(
                event_source=ChangeEventSource.KNOWLEDGEBASE,
                source_id=knowledgebase.id,
                event_type=ChangeEventType.ADD
            )
            logger.info("Default chat doc knowledgebase added to database.")
        except IntegrityError as e:
            logger.error(f"IntegrityError occurred when add default chat doc kb: {e.orig}")
            await session.rollback()

    @with_async_db_session
    async def create_default_embedding_model(self, session: AsyncSession):
        sql_results = await session.exec(select(EmbeddingModelEntity).where(EmbeddingModelEntity.model_id == "BAAI/bge-m3"))
        embedding_entities: List[EmbeddingModelEntity] = sql_results.all()
        if len(embedding_entities) > 0:
            logger.info("Default embedding model already exists.")
            if not embedding_entities[0].is_ready:
                import app.worker as background_worker
                background_worker.download_model.delay(id=embedding_entities[0].id, model_name=embedding_entities[0].model_name)
            return
        logger.info("Creating default embedding model.")
        embedding_model = EmbeddingModelCreate(
            model_id="BAAI/bge-m3",
            type=EmbeddingType.LOCAL,
            dimension=1024,
            embed_batch_size=10,
            is_ready=False,
            is_default=True,
        )
        default_embedding_model = EmbeddingModelEntity.model_validate(embedding_model)
        try:
            embedding_provider.add(default_embedding_model)
            session.add(default_embedding_model)
            await session.commit()
            await session.refresh(default_embedding_model)
            await self.notify_change_async(
                event_source=ChangeEventSource.EMBEDDING,
                source_id=default_embedding_model.id,
                event_type=ChangeEventType.ADD
            )
            logger.info("Default embedding model added to database. Starting worker to download model...")
            import app.worker as worker
            worker.download_model.delay(id=default_embedding_model.id, model_name=default_embedding_model.model_name)
        except IntegrityError as e:
            logger.error(f"IntegrityError occurred when add embedding: {e.orig}")
            await session.rollback()

    @with_async_db_session
    async def create_builtin_gaia_dataset(self, session: AsyncSession):
        # create builtin GAIA evaluation entity if not exists
        sql_results = await session.exec(select(DatasetEntity).where(DatasetEntity.name == "GAIA"))
        evaluation_entities: List[DatasetEntity] = sql_results.all()
        if len(evaluation_entities) > 0:
            logger.info("Builtin GAIA dataset already exists.")
            return
        logger.info("Creating builtin GAIA dataset.")
        gaia_dataset = DatasetCreate(
            name="GAIA",
            description="GAIA评估",
            type="built-in"
        )
        gaia_dataset = DatasetEntity.model_validate(gaia_dataset)
        try:
            evaluation_provider.add(gaia_dataset)
            session.add(gaia_dataset)
            await session.commit()
            await session.refresh(gaia_dataset)
            await self.notify_change_async(
                event_source=ChangeEventSource.EVALUATION,
                source_id=gaia_dataset.id,
                event_type=ChangeEventType.ADD
            )
            logger.info("Builtin GAIA evaluation added to database.")
        except IntegrityError as e:
            logger.error(f"IntegrityError occurred when add gaia evaluation: {e.orig}")
            await session.rollback()

        GAIA_DATASET_PATH = "./resources/dataset/gaia/gaia_level_1_27.jsonl"
        file_results = eval_client.load_dataset_from_local_path(file_path=GAIA_DATASET_PATH)
        for line in file_results:
            dataset_entity = DatasetSampleEntity(
                dataset_id=gaia_dataset.id,
                input=line["input"],
                expected_output=line.get("expected_output"),
                eval_metadata=line.get("metadata")
            )
            session.add(dataset_entity)
            await session.commit()
            logger.info(f"Saved file {dataset_entity} successfully.")



    @with_async_db_session
    async def notify_change_async(
        self,
        session: AsyncSession,
        event_source: ChangeEventSource,
        source_id: str,
        event_type: ChangeEventType,
    ):
        event = ChangeEvent(
            source_id=source_id,
            event_type=event_type,
            event_source=event_source,
        )
        logger.info(f"Submitting change event: {event}")

        session.add(event)
        await session.commit()
        logger.info(f"Notified change event: {event}")


    @with_async_db_session
    async def monitor_changes_async(self, session: AsyncSession):
        while True:
            if self.last_change_dt is not None:
                try:
                    event = (await session.exec(
                        select(ChangeEvent)
                        .where(ChangeEvent.created_at > self.last_change_dt)
                        .order_by(ChangeEvent.created_at.asc())
                    )).first()

                    if not event:
                        await asyncio.sleep(10)
                    else:
                        logger.info(f"Found change event {event}.")
                        await self.process_change(event)
                except Exception:
                    logger.error(
                        f"Error when processing changes. Details:{traceback.format_exc()}"
                    )
                finally:
                    if event:
                        logger.info(f"Updated change timestamp from {self.last_change_dt} to {event.created_at}")
                        self.last_change_dt = event.created_at

    @retry(stop=stop_after_attempt(3))
    async def process_change(
        self,
        event: ChangeEvent,
    ):
        config_provider = self._get_config_provider(event.event_source)
        await config_provider.process_event(
            event_type=event.event_type,
            source_id=event.source_id,
        )
        logger.info(f"Applied change event {event.event_type} for {event}.")

    def _get_config_provider(
        self,
        event_source: ChangeEventSource) -> BaseConfigProvider:
        match event_source:
            case ChangeEventSource.EMBEDDING:
                return embedding_provider
            case ChangeEventSource.LLM:
                return llm_provider
            case ChangeEventSource.KNOWLEDGEBASE:
                return knowledgebase_provider
            case ChangeEventSource.MCP:
                from config.providers.mcp_tool_provider import mcp_provider
                return mcp_provider
            case ChangeEventSource.WEBSEARCH:
                from config.providers.websearch_provider import websearch_provider
                return websearch_provider
            case ChangeEventSource.RERANK:
                from config.providers.reranker_provider import reranker_provider
                return reranker_provider
            case ChangeEventSource.CHATBOT:
                from config.providers.chatbot_provider import chatbot_provider
                return chatbot_provider
            case ChangeEventSource.TRACE:
                from config.providers.trace_provider import trace_provider
                return trace_provider
            case ChangeEventSource.GUARDRAIL:
                from config.providers.guardrail_provider import guardrail_provider
                return guardrail_provider
            case ChangeEventSource.EVALUATION:
                from config.providers.evaluation_provider import evaluation_provider
                return evaluation_provider
            case ChangeEventSource.VECTORDB:
                from config.providers.vectordb_provider import vectordb_provider
                return vectordb_provider
            case ChangeEventSource.CODESANDBOX:
                from config.providers.code_sandbox_provider import codesandbox_provider
                return codesandbox_provider
            case ChangeEventSource.CHATDB:
                from config.providers.chatdb_provider import chatdb_provider
                return chatdb_provider
            case _:
                raise ValueError(f"Unknown event source: {event_source}")

config_change_manager = ConfigChangeManager()
