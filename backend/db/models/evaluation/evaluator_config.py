from datetime import datetime, timezone
import uuid
from typing import Any
from copy import deepcopy
from common.encrypt_utils import decrypt_key, encrypt_key
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime, JSON

class EvaluatorConfigCreate(SQLModel):
    name: str = Field(default="")
    type: str = Field(default="") # ExactMatch, LLMJudge
    model_id: str = Field(default="")
    case_sensitive: bool = Field(default=False)
    ignore_punctuation: bool = Field(default=False)
    # All miscellaneous configurations
    extra_parameters: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    # All sensitive parameter values will be stored as encrypted strings in db.
    sensitive_parameters: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))


class EvaluatorConfigEntityBase(EvaluatorConfigCreate):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    dataset_id: str = Field(
        default="",
        foreign_key="pai_dataset.id",
        description="Reference to the evaluation task",
        ondelete="CASCADE",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime)
    )

class EvaluatorConfigEntity(EvaluatorConfigEntityBase, table=True):
    __tablename__ = "pai_evaluator_config"

    @staticmethod
    def from_create_entity(dataset_id: str, evaluator_config_create: EvaluatorConfigCreate):
        ecte = EvaluatorConfigEntity()
        ecte.dataset_id = dataset_id
        ecte.name = evaluator_config_create.name
        ecte.type = evaluator_config_create.type
        ecte.model_id = evaluator_config_create.model_id
        ecte.case_sensitive = evaluator_config_create.case_sensitive
        ecte.ignore_punctuation = evaluator_config_create.ignore_punctuation
        ecte.extra_parameters = deepcopy(evaluator_config_create.extra_parameters)
        # encrypt the sensitive values
        if evaluator_config_create.sensitive_parameters:
            ecte.sensitive_parameters = {}
            for k, v in evaluator_config_create.sensitive_parameters.items():
                ecte.sensitive_parameters[k] = encrypt_key(v)
            ecte._sensitive_data_is_encrypted = True
        return ecte


class EvaluatorConfigRead(EvaluatorConfigEntityBase):
    @staticmethod
    def from_config_entity(config_entity: EvaluatorConfigEntity):
        ecr = EvaluatorConfigRead()
        ecr.id = config_entity.id
        ecr.dataset_id = config_entity.dataset_id
        ecr.created_at = config_entity.created_at
        ecr.updated_at = config_entity.updated_at
        ecr.name = config_entity.name
        ecr.type = config_entity.type
        ecr.model_id = config_entity.model_id
        ecr.case_sensitive = config_entity.case_sensitive
        ecr.ignore_punctuation = config_entity.ignore_punctuation
        ecr.extra_parameters = deepcopy(config_entity.extra_parameters)
        # encrypt the sensitive values
        if config_entity.sensitive_parameters:
            ecr.sensitive_parameters = {}
            for k, v in config_entity.sensitive_parameters.items():
                ecr.sensitive_parameters[k] = decrypt_key(v)

        return ecr
