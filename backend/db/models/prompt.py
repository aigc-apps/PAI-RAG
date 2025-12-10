from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON
from common.chat.prompts import (
    SYNTHESIZE_PROMPT,
    SYSTEM_PROMPT,
    SEARCH_WEB_TOOL_PROMPT,
    PLANNING_TOOL_PROMPT,
    KNOWLEDGEBASE_TOOL_PROMPT,
    ATTACHMENTS_TOOL_PROMPT
)
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional


DEFAULT_PROMPTS = {
    "system_prompt": SYSTEM_PROMPT,
    "search_web_tool_prompt": SEARCH_WEB_TOOL_PROMPT,
    "planning_tool_prompt": PLANNING_TOOL_PROMPT,
    "attachments_tool_prompt": ATTACHMENTS_TOOL_PROMPT,
    "knowledgebase_tool_prompt": KNOWLEDGEBASE_TOOL_PROMPT,
    "without_tools_prompt": SYNTHESIZE_PROMPT,
}


class PromptModel(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    prompts: dict = Field(default=DEFAULT_PROMPTS, sa_column=Column("prompts", JSON))



class PromptModelEntity(PromptModel, table=True):
    __tablename__ = "pai_prompt_config"

    id: str = Field(default="default_prompt_id", primary_key=True)
