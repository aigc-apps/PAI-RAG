from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON
from chat.prompts import (
    WITHOUT_TOOLS_PROMPT,
    SYSTEM_PROMPT,
    SEARCH_WEB_TOOL_PROMPT,
    THINKING_TOOL_PROMPT,
    KNOWLEDGEBASE_TOOL_PROMPT,
    ATTACHMENTS_TOOL_PROMPT
)

DEFAULT_PROMPTS = {
    "system_prompt": SYSTEM_PROMPT,
    "search_web_tool_prompt": SEARCH_WEB_TOOL_PROMPT,
    "thinking_tool_prompt": THINKING_TOOL_PROMPT,
    "attachments_tool_prompt": ATTACHMENTS_TOOL_PROMPT,
    "knowledgebase_tool_prompt": KNOWLEDGEBASE_TOOL_PROMPT,
    "without_tools_prompt": WITHOUT_TOOLS_PROMPT,
}


class PromptModel(SQLModel):
    prompts: dict = Field(default=DEFAULT_PROMPTS, sa_column=Column("prompts", JSON))



class PromptModelEntity(PromptModel, table=True):
    __tablename__ = "pai_prompt_config"

    id: str = Field(default="default_prompt_id", primary_key=True)
