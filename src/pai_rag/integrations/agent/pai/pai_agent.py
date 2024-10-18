from pydantic import BaseModel
from typing import (
    Any,
    Dict,
    List,
    Callable,
    Optional,
    Type,
)
from llama_index.agent.openai.step import OpenAIAgentWorker
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.callbacks import CallbackManager
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool
from llama_index.llms.openai.utils import OpenAIToolCall
from pai_rag.integrations.agent.pai.utils.tool_utils import (
    get_customized_tools,
)

import logging

logger = logging.getLogger(__name__)

DEFAULT_MAX_FUNCTION_CALLS = 5


# TODO: move to each integration
class AgentConfig(BaseModel):
    tool_definition_file: str | None = None
    python_script_file: str | None = None


def get_tools(agent_config: AgentConfig):
    tools = []
    # tools.extend(get_calculator_tools())
    # logger.info(f"Loaded calculator tools.")
    tools.extend(get_customized_tools(agent_config))
    logger.info("Loaded custom tools.")

    return tools


class PaiAgent(AgentRunner):
    """Pai agent.

    Subclasses AgentRunner with a OpenAIAgentWorker.

    """

    def __init__(
        self,
        tools: List[BaseTool],
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        default_tool_choice: str = "auto",
        callback_manager: Optional[CallbackManager] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        tool_call_parser: Optional[Callable[[OpenAIToolCall], Dict]] = None,
    ) -> None:
        """Init params."""
        callback_manager = callback_manager or llm.callback_manager
        step_engine = OpenAIAgentWorker.from_tools(
            tools=tools,
            tool_retriever=tool_retriever,
            llm=llm,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
            prefix_messages=prefix_messages,
            tool_call_parser=tool_call_parser,
        )
        super().__init__(
            step_engine,
            memory=memory,
            llm=llm,
            callback_manager=callback_manager,
            default_tool_choice=default_tool_choice,
        )

    @classmethod
    def from_tools(
        cls,
        agent_config: AgentConfig,
        tools: Optional[List[BaseTool]] = None,
        tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
        llm: Optional[LLM] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        verbose: bool = False,
        max_function_calls: int = DEFAULT_MAX_FUNCTION_CALLS,
        default_tool_choice: str = "auto",
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        tool_call_parser: Optional[Callable[[OpenAIToolCall], Dict]] = None,
        **kwargs: Any,
    ) -> "PaiAgent":
        """Create an PaiAgent from a list of tools.

        Similar to `from_defaults` in other classes, this method will
        infer defaults for a variety of parameters, including the LLM,
        if they are not specified.

        """
        tools = tools or get_tools(agent_config)

        chat_history = chat_history or []
        llm = llm or Settings.llm
        if callback_manager is not None:
            llm.callback_manager = callback_manager

        memory = memory or memory_cls.from_defaults(chat_history, llm=llm)

        if not llm.metadata.is_function_calling_model:
            raise ValueError(
                f"Model name {llm.model} does not support function calling API. "
            )

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [ChatMessage(content=system_prompt, role="system")]

        prefix_messages = prefix_messages or []

        return cls(
            tools=tools,
            tool_retriever=tool_retriever,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            verbose=verbose,
            max_function_calls=max_function_calls,
            callback_manager=callback_manager,
            default_tool_choice=default_tool_choice,
            tool_call_parser=tool_call_parser,
        )
