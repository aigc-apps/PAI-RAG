import uuid
from sqlmodel import Field, SQLModel, Column, JSON
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional, List


class Skill(SQLModel):
    """Base skill model with common fields."""
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    name: str = Field(default=None)
    description: Optional[str] = Field(default=None)
    # skill 类型: declarative (纯.md), composite (含附加文件), command (含终端命令)
    skill_type: str = Field(default="declarative")
    # 是否启用
    enabled: bool = Field(default=True)
    # skill 的 .md 文件内容（核心指令）
    content: Optional[str] = Field(default=None)
    # YAML frontmatter 中解析出的元数据 (JSON)
    metadata_json: Optional[str] = Field(default=None)
    # skill 需要的工具名列表（JSON 数组字符串）
    required_tools: Optional[str] = Field(default=None)
    # skill 需要的环境变量（JSON 数组字符串）
    required_env: Optional[str] = Field(default=None)
    # skill 前置条件描述
    prerequisites: Optional[str] = Field(default=None)
    # skill 文件夹路径（如果是复合 skill）
    folder_path: Optional[str] = Field(default=None)


class SkillCreate(SQLModel):
    """Model for creating a skill (used in API request)."""
    name: str
    description: Optional[str] = None
    skill_type: str = "declarative"
    enabled: bool = True
    content: Optional[str] = None
    metadata_json: Optional[str] = None
    required_tools: Optional[str] = None
    required_env: Optional[str] = None
    prerequisites: Optional[str] = None
    folder_path: Optional[str] = None


class SkillRead(Skill):
    """Model for reading a skill (used in API response)."""
    id: str = Field(default=None)


class SkillUpdate(SQLModel):
    """Model for updating a skill."""
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    content: Optional[str] = None
    metadata_json: Optional[str] = None
    required_tools: Optional[str] = None
    required_env: Optional[str] = None
    prerequisites: Optional[str] = None


# Database table entity
class SkillEntity(Skill, table=True):
    __tablename__ = "pai_skill"

    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), primary_key=True)
