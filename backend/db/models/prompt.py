from sqlmodel import Field, SQLModel
from chat.prompts import (
    WITHOUT_TOOLS_PROMPT,
    SYSTEM_PROMPT,
    SEARCH_WEB_TOOL_PROMPT,
    THINKING_TOOL_PROMPT,
    KNOWLEDGEBASE_TOOL_PROMPT,
    ATTACHMENTS_TOOL_PROMPT
)


class PromptModel(SQLModel):
    system_prompt: str = Field(default=SYSTEM_PROMPT)
    search_web_tool_prompt: str = Field(default=SEARCH_WEB_TOOL_PROMPT)
    thinking_tool_prompt: str = Field(default=THINKING_TOOL_PROMPT)
    attachments_tool_prompt: str = Field(default=ATTACHMENTS_TOOL_PROMPT)
    knowledgebase_tool_prompt: str = Field(default=KNOWLEDGEBASE_TOOL_PROMPT)
    without_tools_prompt: str = Field(default=WITHOUT_TOOLS_PROMPT)



class PromptModelEntity(PromptModel, table=True):
    __tablename__ = "pai_prompt_config"

    id: str = Field(default="default_prompt_id", primary_key=True)
