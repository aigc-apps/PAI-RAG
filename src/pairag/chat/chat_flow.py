import time
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Sequence,
)
from llama_index.core.schema import NodeWithScore
from pairag.core.rag_config import RagConfig
from pairag.core.rag_module import (
    resolve_huggingface_embedding,
    resolve_chat_llm,
    resolve_db_retriever,
    resolve_llm_guardrail,
    resolve_openai_query_transform,
    resolve_searcher,
    resolve_news_tool,
    resolve_synthesizer,
    resolve_vector_index,
    resolve_index_retriever_from_retrieval_settings,
    resolve_postprocessor_from_retrieval_settings,
)
from pairag.chat.utils.chat_utils import (
    chat_id_generator,
    make_completion_chunk_response,
    make_completion_response,
    response_gen_from_text,
    response_from_text,
)

from pairag.integrations.query_transform.pai_query_transform import (
    messages_to_history_str,
)
from pairag.integrations.query_transform.intent_models import (
    ChatIntentType,
    ChatToolType,
    IntentResult,
)
from pairag.knowledgebase.rag_knowledgebase import knowledgebase_manager
from pairag.chat.models import (
    ChatCompletionRequest,
    ChatResponseWrapper,
    EmbeddingInput,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ImageBlock,
)

from openai.types.chat import (
    ChatCompletion,
)
from openai.types.embedding import Embedding
from openai.types.create_embedding_response import (
    CreateEmbeddingResponse,
    Usage as EmbeddingUsage,
)

from loguru import logger

from pairag.utils.time_utils import get_prompt_current_time_str
from pairag.integrations.trace.pai_query_wrapper import pai_query_wrapper
import llama_index.core.instrumentation as instrument
from pairag.chat.utils.message_utils import (
    remove_think_from_messages,
    message_is_empty,
    parse_system_prompt,
    parse_messages,
)


dispatcher = instrument.get_dispatcher(__name__)

DEFAULT_GUARDRAIL_RESPONSE = "抱歉，无法处理这个请求。"
DEFAULT_EMPTY_RESPONSE = "看起来你发了一条空白消息，有什么能帮到你的吗？"
DEFAULT_ERROR_RESPONSE = "抱歉，系统出错，暂时无法处理这个请求。"


class ChatFlow:
    def __init__(self, config: RagConfig):
        self.config = config

    @dispatcher.span
    async def _rewrite_query(
        self,
        chat_request: ChatCompletionRequest,
        chat_history_str: str,
    ) -> IntentResult:
        if chat_request.intent == ChatIntentType.CHAT_LLM:
            return IntentResult(
                intent=ChatIntentType.CHAT_LLM,
                query=chat_request.messages[-1].content,
            )
        else:
            query_transform = resolve_openai_query_transform(self.config)
            if query_transform is not None:
                intent_result = await query_transform.arewrite(
                    intent=chat_request.intent,
                    chat_messages=chat_request.messages,
                    chat_history_str=chat_history_str,
                )
                return intent_result

            else:
                logger.info("No query transform found, using default intent.")
                return IntentResult(
                    intent=chat_request.intent,
                    query_str=chat_request.messages[-1].content,
                )

    @dispatcher.span
    async def _recognize_intent(
        self,
        chat_request: ChatCompletionRequest,
        chat_history_str: str,
    ) -> IntentResult:
        # 默认RAG
        potential_intents = [ChatToolType.CHAT_LLM]

        tool_switches = {
            ChatToolType.CHAT_KNOWLEDGEBASE: chat_request.chat_knowledgebase
            or chat_request.index_name,
            ChatToolType.SEARCH_WEB: chat_request.search_web,
            ChatToolType.CHAT_DB: chat_request.chat_db,
            ChatToolType.CHAT_NEWS: chat_request.chat_news,
        }
        enabled_tools = [k for k, v in tool_switches.items() if v]
        potential_intents.extend(enabled_tools)

        query_transform = resolve_openai_query_transform(self.config)
        logger.debug(
            f"[Parameters][QueryTransform] {query_transform}, [potential_intents]{potential_intents}"
        )

        if query_transform is not None and len(potential_intents) > 1:
            intent_result = await query_transform.arun(
                chat_messages=chat_request.messages,
                potential_intents=potential_intents,
                chat_history_str=chat_history_str,
            )
            return intent_result

        else:
            logger.info(
                f"No query transform found, using default intent. {potential_intents[-1].value}"
            )
            return IntentResult(
                intent=potential_intents[-1],
                query_str=chat_request.messages[-1].content,
            )

    @pai_query_wrapper()
    async def astream_chat(
        self,
        chat_request: ChatCompletionRequest,
    ) -> AsyncGenerator[str, None]:
        logger.debug(f"Streaming chat request: {chat_request}")
        start_time = time.time()
        chat_id = chat_id_generator()
        response_wrapper = await self._achat_internal(
            chat_id=chat_id,
            chat_request=chat_request,
            start_time=start_time,
        )
        return make_completion_chunk_response(
            chat_id=chat_id,
            model=chat_request.model,
            response_wrapper=response_wrapper,
            start_time=start_time,
            return_reference=chat_request.return_reference,
        )

    @pai_query_wrapper()
    async def achat(
        self,
        chat_request: ChatCompletionRequest,
    ) -> ChatCompletion:
        start_time = time.time()
        chat_id = chat_id_generator()
        response_wrapper = await self._achat_internal(
            chat_id=chat_id,
            chat_request=chat_request,
            start_time=start_time,
        )
        return make_completion_response(
            chat_id=chat_id,
            model=chat_request.model,
            response_wrapper=response_wrapper,
            return_reference=chat_request.return_reference,
        )

    @dispatcher.span
    async def achat_db(
        self,
        query_str: str,
        chat_history_str: str,
        model_id: str,
        stream: bool = False,
        **llm_kwargs,
    ):
        db_retriever = resolve_db_retriever(self.config, model_id=model_id)
        if not db_retriever:
            raise ValueError(
                "DBChat config is not valid. Please check your DBChat api configuration."
            )

        nodes = await db_retriever.aretrieve(query_str)

        db_schema = ""
        query_code_instruction = ""
        if len(nodes) > 0:
            db_schema = nodes[0].node.metadata.get("db_schema", "")
            query_code_instruction = nodes[0].node.metadata.get(
                "query_code_instruction", ""
            )

        synthesizer = resolve_synthesizer(self.config, model_id=model_id)

        response = await synthesizer.asynthesize(
            query_str=query_str,
            nodes=nodes,
            stream=stream,
            chat_history_str=chat_history_str,
            system_role_str=self.config.data_analysis.system_role_prompt,
            prompt_template_str=self.config.data_analysis.synthesizer_prompt,
            prompt_template_args={
                "db_schema": db_schema,
                "query_code_instruction": query_code_instruction,
            },
            **llm_kwargs,
        )

        return response

    @dispatcher.span
    async def achat_news(
        self,
        query_str: str,
        stream: bool = True,
    ) -> ChatResponseWrapper:
        news_tool = resolve_news_tool(self.config)
        if not news_tool:
            raise ValueError("抱歉，无法查询新闻信息。请检查新闻工具配置。")

        if stream:
            response_gen = await news_tool.astream_chat(messages=[], prompt=query_str)
            response_wrapper = ChatResponseWrapper(response=response_gen)
        else:
            response = await news_tool.achat(messages=[], prompt=query_str)
            response_wrapper = ChatResponseWrapper(response=response)

        return response_wrapper

    @dispatcher.span
    async def achat_news_llm(
        self,
        query_str: str,
        stream: bool = True,
    ) -> ChatResponseWrapper:
        news_tool = resolve_news_tool(self.config)
        if not news_tool:
            raise ValueError("抱歉，无法查询新闻信息。请检查新闻工具配置。")

        if not stream:
            response_wrapper = await news_tool.achat_llm(query_str=query_str)
        else:
            response_wrapper = await news_tool.astream_chat_llm(query_str=query_str)

        return response_wrapper

    @dispatcher.span
    async def achat_web(
        self,
        query_str: str,
        original_user_message: str,
        chat_history_str: str = None,
        model_id: str = None,
        stream: bool = False,
        **lm_kwargs,
    ) -> ChatResponseWrapper:
        search_engine = resolve_searcher(self.config, model_id=model_id)
        if not search_engine:
            raise ValueError(
                "Web search config is not valid. Please check your search api configuration."
            )
        nodes = await search_engine.aretrieve(query_str)

        synthesizer = resolve_synthesizer(self.config, model_id=model_id)

        response = await synthesizer.asynthesize(
            query_str=original_user_message,  # 不使用改写查询生成答案
            nodes=nodes,
            stream=stream,
            chat_history_str=chat_history_str,
            system_role_str=self.config.search.search_role_template,
            prompt_template_str=self.config.search.search_qa_prompt_template,
            **lm_kwargs,
        )
        return response

    @dispatcher.span
    async def achat_knowledgebase(
        self,
        query_str: str,
        original_user_message: str,
        image_blocks: Sequence[ImageBlock] = [],
        knowledgebase_name: str = "default",
        chat_history_str: str = None,
        model_id: str = None,
        stream: bool = False,
        **lm_kwargs,
    ) -> ChatResponseWrapper:
        nodes = await self.aretrieve(
            query_str=query_str, knowledgebase_name=knowledgebase_name
        )
        knowledgebase = knowledgebase_manager.get_knowledgebase(knowledgebase_name)
        synthesizer = resolve_synthesizer(self.config, model_id=model_id)

        qa_prompt_templates = knowledgebase.qa_prompt_templates
        response = await synthesizer.asynthesize(
            query_str=original_user_message,
            nodes=nodes,
            image_blocks=image_blocks,
            stream=stream,
            chat_history_str=chat_history_str,
            system_role_str=qa_prompt_templates["system_prompt_template"],
            prompt_template_str=qa_prompt_templates["task_prompt_template"],
            **lm_kwargs,
        )

        return response

    @dispatcher.span
    async def achat_llm(
        self,
        model_id: str,
        messages: List[ChatMessage],
        system_prompt: str = None,
        stream: bool = False,
        **llm_kwargs,
    ) -> ChatResponseWrapper:
        llm = resolve_chat_llm(self.config, model_id=model_id)
        system_role = system_prompt or self.config.synthesizer.system_role_template

        prompt_messages = []
        current_datetime = get_prompt_current_time_str()

        system_prompt = self.config.synthesizer.custom_prompt_template.format(
            current_datetime=current_datetime,
            cur_date=current_datetime,
        )

        if system_role:
            system_prompt = f"{system_role}\n\n{system_prompt}"

        prompt_messages.append(
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
        )

        messages = prompt_messages + messages[-7:]

        if stream:
            response_gen = await llm.astream_chat(messages, **llm_kwargs)
            return ChatResponseWrapper(response=response_gen)
        else:
            response = await llm.achat(messages, **llm_kwargs)
            return ChatResponseWrapper(response=response)

    async def aretrieve(
        self,
        query_str: str,
        knowledgebase_name: str = None,
        extra_retrieval_settings: Dict[str, Any] = {},
    ) -> List[NodeWithScore]:
        knowledgebase = knowledgebase_manager.get_knowledgebase(knowledgebase_name)
        _retrieval_settings = {
            **knowledgebase.retrieval_settings,
            **extra_retrieval_settings,
        }
        logger.info(
            f"Retrieving {knowledgebase_name} with query {query_str}, retrieve settings: {_retrieval_settings}."
        )

        vector_index = resolve_vector_index(knowledgebase=knowledgebase)

        retriever = resolve_index_retriever_from_retrieval_settings(
            vector_index=vector_index,
            retrieval_settings=_retrieval_settings,
        )

        postprocessor = resolve_postprocessor_from_retrieval_settings(
            retrieval_settings=_retrieval_settings
        )
        nodes = await retriever.aretrieve(query_str)

        reranked_nodes = await postprocessor.apostprocess_nodes(
            nodes,
            query_str=query_str,
        )
        return reranked_nodes

    async def _achat_internal(
        self,
        chat_id: str,
        chat_request: ChatCompletionRequest,
        start_time: float = 0.0,
    ) -> ChatResponseWrapper:
        system_prompt, messages = parse_system_prompt(chat_request.messages)
        messages = parse_messages(messages)
        if message_is_empty(messages):
            if chat_request.stream:
                return response_gen_from_text(DEFAULT_EMPTY_RESPONSE)
            return response_from_text(DEFAULT_EMPTY_RESPONSE)

        chat_request.messages = remove_think_from_messages(messages)
        image_blocks = [
            block for block in messages[-1].blocks if isinstance(block, ImageBlock)
        ]

        chat_history_str = messages_to_history_str(chat_request.messages[-7:-1])

        original_user_message = chat_request.messages[-1].content
        llm_kwargs = self._get_llm_kwargs(chat_request=chat_request)
        if chat_request.intent is None:
            # 意图识别
            intent_result = await self._recognize_intent(
                chat_request, chat_history_str=chat_history_str
            )
        else:
            # 直接改写
            intent_result = await self._rewrite_query(
                chat_request, chat_history_str=chat_history_str
            )

        logger.info(
            f"[{chat_id}] Intent recognized: {intent_result.intent}, query: {intent_result.query_str}, elapsed time: {time.time() - start_time}s. Token usage: {intent_result.token_usage}"
        )

        # 安全护栏
        guardrail = resolve_llm_guardrail(self.config)
        if guardrail is not None:
            check_result = await guardrail.acheck(text=original_user_message)
            if check_result.reject:
                logger.info(f"Guadrail check failed: {original_user_message}.")
                if chat_request.stream:
                    return response_gen_from_text(check_result.advice)
                else:
                    return response_from_text(check_result.advice)

            logger.info(f"Guadrail check passed: {original_user_message}.")

        # 意图分发
        logger.info(f"Routing query {original_user_message} to {intent_result.intent}")
        if intent_result.intent == ChatIntentType.CHAT_LLM:
            response_wrapper = await self.achat_llm(
                model_id=chat_request.model,
                messages=chat_request.messages,
                stream=chat_request.stream,
                system_prompt=system_prompt,
                **llm_kwargs,
            )
        elif intent_result.intent == ChatIntentType.CHAT_NEWS:
            response_wrapper = await self.achat_news(
                query_str=intent_result.query_str,
                stream=chat_request.stream,
            )
        elif intent_result.intent == ChatIntentType.CHAT_NEWS_LLM:
            response_wrapper = await self.achat_news_llm(
                query_str=intent_result.query_str,
                stream=chat_request.stream,
            )
        elif intent_result.intent == ChatIntentType.SEARCH_WEB:
            response_wrapper = await self.achat_web(
                query_str=intent_result.query_str,
                original_user_message=original_user_message,
                chat_history_str=chat_history_str,
                stream=chat_request.stream,
                model_id=chat_request.model,
                **llm_kwargs,
            )
        elif intent_result.intent == ChatIntentType.CHAT_DB:
            response_wrapper = await self.achat_db(
                query_str=intent_result.query_str,
                chat_history_str=chat_history_str,
                stream=chat_request.stream,
                model_id=chat_request.model,
                **llm_kwargs,
            )
        elif intent_result.intent == ChatIntentType.CHAT_KNOWLEDGEBASE:
            response_wrapper = await self.achat_knowledgebase(
                query_str=intent_result.query_str,
                original_user_message=original_user_message,
                image_blocks=image_blocks,
                chat_history_str=chat_history_str,
                stream=chat_request.stream,
                model_id=chat_request.model,
                knowledgebase_name=chat_request.index_name,
                **llm_kwargs,
            )
        else:
            logger.warning(f"Unknown intent: {intent_result.intent}")
            response_wrapper = await self.achat_llm(
                model_id=chat_request.model,
                messages=chat_request.messages,
                stream=chat_request.stream,
                system_prompt=system_prompt,
                **llm_kwargs,
            )

        response_wrapper.intent_result = intent_result
        return response_wrapper

    def _get_llm_kwargs(self, chat_request: ChatCompletionRequest) -> Dict:
        llm_kwargs = {}
        if chat_request.temperature is not None:
            llm_kwargs["temperature"] = chat_request.temperature
        if chat_request.max_tokens is not None:
            llm_kwargs["max_tokens"] = chat_request.max_tokens

        return llm_kwargs

    # 原子能力

    @dispatcher.span
    async def achat_llm_atomic(
        self,
        chat_request: ChatCompletionRequest,
    ) -> ChatResponseWrapper:
        logger.info(f"achat_llm_atomic: {chat_request}")
        chat_id = chat_id_generator()

        llm = resolve_chat_llm(self.config, model_id=chat_request.model)
        llm_kwargs = self._get_llm_kwargs(chat_request)
        response = await llm.achat(chat_request.messages, **llm_kwargs)
        response_wrapper = ChatResponseWrapper(response=response)
        return make_completion_response(
            chat_id=chat_id,
            model=chat_request.model,
            response_wrapper=response_wrapper,
            return_reference=False,
        )

    @pai_query_wrapper()
    async def astream_llm_atomic(
        self,
        chat_request: ChatCompletionRequest,
    ) -> ChatResponseWrapper:
        logger.info(f"astream_llm_atomic: {chat_request}")
        start_time = time.time()
        chat_id = chat_id_generator()

        llm = resolve_chat_llm(self.config, model_id=chat_request.model)
        llm_kwargs = self._get_llm_kwargs(chat_request)
        response = await llm.astream_chat(chat_request.messages, **llm_kwargs)
        response_wrapper = ChatResponseWrapper(response=response)
        response_wrapper.intent_result = chat_request.intent

        return make_completion_chunk_response(
            chat_id=chat_id,
            model=chat_request.model,
            response_wrapper=response_wrapper,
            start_time=start_time,
            return_reference=False,
        )

    @dispatcher.span
    async def achat_web_atomic(
        self,
        chat_request: ChatCompletionRequest,
    ) -> ChatResponseWrapper:
        logger.info(f"achat_web_atomic: {chat_request}")
        chat_id = chat_id_generator()

        _, messages = parse_system_prompt(chat_request.messages)
        messages = parse_messages(messages)
        if message_is_empty(messages):
            if chat_request.stream:
                return response_gen_from_text(DEFAULT_EMPTY_RESPONSE)
            return response_from_text(DEFAULT_EMPTY_RESPONSE)

        messages = remove_think_from_messages(messages)
        chat_history_str = messages_to_history_str(messages[-7:-1])
        llm_kwargs = self._get_llm_kwargs(chat_request)

        response_wrapper = await self.achat_web(
            query_str=chat_request.intent.query_str or messages.messages[-1].content,
            original_user_message=messages[-1].content,
            chat_history_str=chat_history_str,
            stream=chat_request.stream,
            model_id=chat_request.model,
            **llm_kwargs,
        )

        response_wrapper.intent_result = chat_request.intent
        return make_completion_response(
            chat_id=chat_id,
            model=chat_request.model,
            response_wrapper=response_wrapper,
            return_reference=False,
        )

    @pai_query_wrapper()
    async def astream_web_atomic(
        self,
        chat_request: ChatCompletionRequest,
    ) -> ChatResponseWrapper:
        logger.info(f"astream_web_atomic: {chat_request}")
        start_time = time.time()
        chat_id = chat_id_generator()

        _, messages = parse_system_prompt(chat_request.messages)
        messages = parse_messages(messages)
        if message_is_empty(messages):
            if chat_request.stream:
                return response_gen_from_text(DEFAULT_EMPTY_RESPONSE)
            return response_from_text(DEFAULT_EMPTY_RESPONSE)

        messages = remove_think_from_messages(messages)
        chat_history_str = messages_to_history_str(messages[-7:-1])
        llm_kwargs = self._get_llm_kwargs(chat_request)

        response_wrapper = await self.achat_web(
            query_str=chat_request.intent.query_str or messages[-1].content,
            original_user_message=messages[-1].content,
            chat_history_str=chat_history_str,
            stream=chat_request.stream,
            model_id=chat_request.model,
            **llm_kwargs,
        )

        response_wrapper.intent_result = chat_request.intent
        return make_completion_chunk_response(
            chat_id=chat_id,
            model=chat_request.model,
            response_wrapper=response_wrapper,
            start_time=start_time,
            return_reference=False,
        )

    @dispatcher.span
    async def achat_knowledgebase_atomic(
        self,
        chat_request: ChatCompletionRequest,
    ) -> ChatResponseWrapper:
        logger.info(f"achat_knowledgebase_atomic: {chat_request}")
        chat_id = chat_id_generator()

        _, messages = parse_system_prompt(chat_request.messages)
        messages = parse_messages(messages)
        if message_is_empty(messages):
            if chat_request.stream:
                return response_gen_from_text(DEFAULT_EMPTY_RESPONSE)
            return response_from_text(DEFAULT_EMPTY_RESPONSE)

        messages = remove_think_from_messages(messages)
        chat_history_str = messages_to_history_str(messages[-7:-1])
        llm_kwargs = self._get_llm_kwargs(chat_request)

        response_wrapper = await self.achat_knowledgebase(
            query_str=chat_request.intent.query_str or messages[-1].content,
            original_user_message=messages[-1].content,
            chat_history_str=chat_history_str,
            knowledgebase_name=chat_request.index_name,
            stream=chat_request.stream,
            model_id=chat_request.model,
            **llm_kwargs,
        )

        response_wrapper.intent_result = chat_request.intent
        return make_completion_response(
            chat_id=chat_id,
            model=chat_request.model,
            response_wrapper=response_wrapper,
            return_reference=False,
        )

    @pai_query_wrapper()
    async def astream_knowledgebase_atomic(
        self,
        chat_request: ChatCompletionRequest,
    ) -> ChatResponseWrapper:
        logger.info(f"astream_knowledgebase_atomic: {chat_request}")
        start_time = time.time()
        chat_id = chat_id_generator()

        _, messages = parse_system_prompt(chat_request.messages)
        messages = parse_messages(messages)
        if message_is_empty(messages):
            if chat_request.stream:
                return response_gen_from_text(DEFAULT_EMPTY_RESPONSE)
            return response_from_text(DEFAULT_EMPTY_RESPONSE)

        messages = remove_think_from_messages(messages)
        chat_history_str = messages_to_history_str(messages[-7:-1])
        llm_kwargs = self._get_llm_kwargs(chat_request)

        response_wrapper = await self.achat_knowledgebase(
            query_str=chat_request.intent.query_str or messages[-1].content,
            original_user_message=messages[-1].content,
            chat_history_str=chat_history_str,
            knowledgebase_name=chat_request.index_name,
            stream=chat_request.stream,
            model_id=chat_request.model,
            **llm_kwargs,
        )

        response_wrapper.intent_result = chat_request.intent
        return make_completion_chunk_response(
            chat_id=chat_id,
            model=chat_request.model,
            response_wrapper=response_wrapper,
            start_time=start_time,
            return_reference=False,
        )

    @dispatcher.span
    async def arecognize_intent(
        self,
        chat_request: ChatCompletionRequest,
    ) -> IntentResult:
        logger.info(f"arecognize_intent: {chat_request}")
        _, messages = parse_system_prompt(chat_request.messages)
        messages = parse_messages(messages)
        chat_request.messages = remove_think_from_messages(messages)
        chat_history_str = messages_to_history_str(chat_request.messages[-7:-1])

        # 意图识别
        intent_result = await self._recognize_intent(
            chat_request, chat_history_str=chat_history_str
        )

        return intent_result

    @dispatcher.span
    async def aembed(
        self,
        embedding_input: EmbeddingInput,
    ) -> CreateEmbeddingResponse:
        assert embedding_input.input is not None, "embeddig 'input' cannot be None"

        text_inputs = []
        if isinstance(embedding_input.input, str):
            text_inputs = [embedding_input.input]
        elif isinstance(embedding_input.input, list):
            assert (
                len(embedding_input.input) > 0
            ), "embeddig 'input' cannot be empty list."
            assert all(
                item is not None and isinstance(item, str)
                for item in embedding_input.input
            ), "embedding 'input' must be a list of strings."
            text_inputs = embedding_input.input
        else:
            raise ValueError("embedding 'input' must be a string or a list of strings.")

        logger.info(f"aembed: {embedding_input}.")

        embed_model = resolve_huggingface_embedding(model=embedding_input.model)
        text_embeddings = await embed_model.aget_text_embedding_batch(text_inputs)
        embedding_data_list = [
            Embedding(
                embedding=embedding,
                index=i,
                object="embedding",
            )
            for i, embedding in enumerate(text_embeddings)
        ]
        logger.info(f"aembed: finished embedding {len(embedding_data_list)} texts.")
        return CreateEmbeddingResponse(
            object="list",
            data=embedding_data_list,
            model=embedding_input.model,
            usage=EmbeddingUsage(
                prompt_tokens=0,
                total_tokens=0,
            ),
        )
