from typing import List
from chat.agent.models import PlanOutput
from pydantic import BaseModel, Field
from common.chat.constants import MessageRole
from utils.time_utils import get_current_time_str
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall


def get_message_content(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        text = ""
        for item in content:
            if item.get("type") == "text":
                text += item.get("text", "")
        return text

    return content


class AgentState(BaseModel):
    messages: List[dict]
    chat_history: str
    user_query: str
    context_variables: dict = Field(default_factory=dict) # user id / session_id / current_time
    should_stop: bool = Field(default=False)
    step: int = Field(default=1)
    plan: PlanOutput = None
    observations: str = ""
    enable_agent: bool = Field(default=False)
    current_tool_call: ChoiceDeltaToolCall = None

    @classmethod
    def from_messages(
        cls,
        messages: List[dict],
        enable_agent: bool = False,
        context_variables: dict = {},
        plan: PlanOutput = {"steps": []},
    ):
        user_query = ""
        chat_history = ""
        context_variables["current_datetime"] = get_current_time_str()

        filtered_messages = []
        for message in messages:
            content = get_message_content(message)
            if message["role"] == MessageRole.USER:
                filtered_messages.append(message)
                user_query = content
                chat_history += f"{message['role']}: {content}\n"
            elif message["role"] == MessageRole.ASSISTANT:
                chat_history += f"{message['role']}: {content}\n"
                msg_content = message.get("content", "")
                if isinstance(msg_content, list):
                    content = []
                    for item in msg_content:
                        if item.get("type") == "text":
                            content.append(item)

                    if len(content) > 0:
                        filtered_messages.append({"role": MessageRole.ASSISTANT, "content": content})

                else:
                    filtered_messages.append({"role": MessageRole.ASSISTANT, "content": msg_content})

        return cls(
            messages=filtered_messages,
            user_query=user_query,
            chat_history=chat_history,
            context_variables=context_variables,
            step=1,
            plan=plan,
            enable_agent=enable_agent,
        )


    def format_context_str(self) -> str:
        context_str = "## Context info\n\n" + "\n".join([f"{k}: {v}" for k, v in self.context_variables.items()])
        return context_str
