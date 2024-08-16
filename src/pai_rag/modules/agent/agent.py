import logging
from typing import Dict, List, Any
from llama_index.core.llms import ChatMessage
from llama_index.core.agent.react import ReActAgent
from pai_rag.integrations.agent.function_calling.step import (
    MyFunctionCallingAgentWorker,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

logger = logging.getLogger(__name__)


class AgentModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return [
            "LlmModule",
            "FunctionCallingLlmModule",
            "CustomConfigModule",
            "ToolModule",
        ]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        func_tool = new_params["ToolModule"]
        agent_config, _ = new_params["CustomConfigModule"]
        type = config["type"].lower()
        if type == "react":
            llm = new_params["LlmModule"]
            logger.info(
                f"""
                [Parameters][Agent]
                    type = {config.get("type", "Unknown agent type")}
                """
            )
            agent = ReActAgent.from_tools(tools=func_tool, llm=llm, verbose=True)
        elif type == "function_calling":
            llm = new_params["FunctionCallingLlmModule"]
            print("FunctionCallingLlmModule llm", llm)
            if agent_config:
                system_content = agent_config["agent"]["system_prompt"]
            else:
                system_content = config["system_prompt"]
            logger.info(
                f"""
                [Parameters][Agent]
                    type = {type}
                    system_prompt = {system_content}
                """
            )
            prefix_messages = [
                ChatMessage(
                    role="system",
                    content=(system_content),
                )
            ]
            worker = MyFunctionCallingAgentWorker(
                tools=func_tool,
                llm=llm,
                max_function_calls=10,
                prefix_messages=prefix_messages,
                allow_parallel_tool_calls=False,
                verbose=True,
            )
            agent = worker.as_agent()
        else:
            raise ValueError(f"Unknown Agent type: '{config['type']}'")

        return agent
