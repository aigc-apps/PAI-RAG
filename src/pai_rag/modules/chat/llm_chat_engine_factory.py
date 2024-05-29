"""chat engine factory based on config"""

import logging
from typing import List, Dict, Any

from llama_index.core.chat_engine import SimpleChatEngine

from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.utils.messages_utils import parse_chat_messages

logger = logging.getLogger(__name__)


class LlmChatEngineFactoryModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["LlmModule", "ChatStoreModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        self.config = new_params[MODULE_PARAM_CONFIG]
        self.llm = new_params["LlmModule"]
        self.chat_store = new_params["ChatStoreModule"]
        return self

    def get_chat_engine(self, session_id, chat_history):
        chat_memory = self.chat_store.get_chat_memory_buffer(session_id)
        if chat_history is not None:
            history_messages = parse_chat_messages(chat_history)
            for hist_mes in history_messages:
                chat_memory.put(hist_mes)

        if self.config.type == "SimpleChatEngine":
            my_chat_engine = SimpleChatEngine.from_defaults(
                llm=self.llm,
                memory=chat_memory,
                verbose=True,
            )
            logger.info("simple chat_engine instance created")
        else:
            raise ValueError(f"Unknown chat_engine_type: {self.config.type}")

        return my_chat_engine
