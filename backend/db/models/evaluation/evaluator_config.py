from datetime import datetime, timezone
import uuid
from typing import Dict
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime, JSON
from common.encrypt_utils import encrypt_key, decrypt_key

class EvaluatorConfigCreate(SQLModel):
    name: str = Field(default="")
    type: str = Field(default="") # ExactMatch, LLMJudge
    model_id: str = Field(default="")
    case_sensitive: bool = Field(default=False)
    ignore_punctuation: bool = Field(default=False)
    extra_params: Dict[str, str] = Field(
        default_factory=dict,
        sa_column=Column("extra_params", JSON, nullable=False)
    )

class EvaluatorConfigEntity(EvaluatorConfigCreate, table=True):
    __tablename__ = "pai_evaluator_config"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    dataset_id: str = Field(
        foreign_key="pai_dataset.id",
        description="Reference to the evaluation task"
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime)
    )

    def encrypt_extra_params(self):
        """encrypt eval_config_entity.extra_params if needed."""
        if self.type in ["ExactMatch", "LLMJudge"]:
            self.extra_params = {}
        if "access_key_id" in self.extra_params:
            self.extra_params["access_key_id"] = encrypt_key(self.extra_params["access_key_id"])
        if "access_key_secret" in self.extra_params:
            self.extra_params["access_key_secret"] = encrypt_key(self.extra_params["access_key_secret"])

    def decrypt_extra_params(self):
        """decrypt eval_config_entity.extra_params if needed."""
        if "access_key_id" in self.extra_params:
            self.extra_params["access_key_id"] = decrypt_key(self.extra_params["access_key_id"])
        if "access_key_secret" in self.extra_params:
            self.extra_params["access_key_secret"] = decrypt_key(self.extra_params["access_key_secret"])
