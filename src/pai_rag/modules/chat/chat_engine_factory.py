"""chat engine factory based on config"""

import logging
from typing import List, Dict, Any

from llama_index.core.chat_engine import (
    CondenseQuestionChatEngine,
)

from pai_rag.utils.prompt_template import (
    CONDENSE_QUESTION_CHAT_ENGINE_PROMPT_ZH,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.utils.messages_utils import parse_chat_messages


logger = logging.getLogger(__name__)


class ChatEngineFactory:
    def __init__(self, chat_type, query_engine, chat_store):
        self.chat_type = chat_type
        self.query_engine = query_engine
        self.chat_store = chat_store

    def get_chat_engine(self, session_id, chat_history):
        chat_memory = self.chat_store.get_chat_memory_buffer(session_id)
        if chat_history is not None:
            history_messages = parse_chat_messages(chat_history)
            for hist_mes in history_messages:
                chat_memory.put(hist_mes)

        if self.chat_type == "CondenseQuestionChatEngine":
            my_chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=self.query_engine,
                condense_question_prompt=CONDENSE_QUESTION_CHAT_ENGINE_PROMPT_ZH,
                memory=chat_memory,
                verbose=True,
            )
            logger.info("condense_question chat_engine instance created")
            return my_chat_engine
        else:
            raise ValueError(f"Unknown chat_engine_type: {self.config.type}")


class ChatEngineFactoryModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["QueryEngineModule", "ChatStoreModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        query_engine = new_params["QueryEngineModule"]
        chat_store = new_params["ChatStoreModule"]

        return ChatEngineFactory(
            config.type, query_engine=query_engine, chat_store=chat_store
        )
