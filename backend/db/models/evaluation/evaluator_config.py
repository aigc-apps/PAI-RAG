from datetime import datetime, timezone
import uuid
from typing import Any
from copy import deepcopy
import json
from common.encrypt_utils import decrypt_key, encrypt_key
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime, JSON

class EvaluatorConfigBase(SQLModel):
    name: str = Field(default="")
    type: str = Field(default="") # ExactMatch, LLMJudge
    model_id: str = Field(default="")
    case_sensitive: bool = Field(default=False)
    ignore_punctuation: bool = Field(default=False)
    # All miscellaneous configurations
    misc_parameters: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class EvaluatorConfigCreate(EvaluatorConfigBase):
    # All sensitive configurations will be stored as an encrypted str in db.
    sensitive_parameters: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class EvaluatorConfigTableBase(EvaluatorConfigBase):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    dataset_id: str = Field(
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


class EvaluatorConfigRead(EvaluatorConfigTableBase):
    sensitive_parameters: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class EvaluatorConfigEntity(EvaluatorConfigTableBase, table=True):
    """id, name, misc, encrypted"""
    __tablename__ = "pai_evaluator_config"

    encrypted_parameters: str = Field(default='', description='encrypted str representation of sensitive parameter dict')

    @staticmethod
    def from_create_entity(dataset_id: str, ecc: EvaluatorConfigCreate):
        ece = EvaluatorConfigEntity()
        ece.dataset_id = dataset_id
        ece.name = ecc.name
        ece.type = ecc.type
        ece.model_id = ecc.model_id
        ece.case_sensitive = ecc.case_sensitive
        ece.ignore_punctuation = ecc.ignore_punctuation
        ece.misc_parameters = deepcopy(ecc.misc_parameters)
        if ecc.sensitive_parameters:
            str_value = json.dumps(ecc.sensitive_parameters)
            encrypted_params = encrypt_key(str_value)
            ece.encrypted_parameters = encrypted_params

        return ece

    def to_read_entity(self):
        """decrypt encrypted params to sensitive params"""
        ecr = EvaluatorConfigRead()
        ecr.id = self.id
        ecr.dataset_id = self.dataset_id
        ecr.name = self.name
        ecr.type = self.type
        ecr.model_id = self.model_id
        ecr.case_sensitive = self.case_sensitive
        ecr.ignore_punctuation = self.ignore_punctuation
        ecr.misc_parameters = deepcopy(self.misc_parameters)
        ecr.created_at = self.created_at
        ecr.updated_at = self.updated_at
        if self.encrypted_parameters:
            decrypted_value = decrypt_key(self.encrypted_parameters)
        else:
            decrypted_value = '{}'

        ecr.sensitive_parameters = json.loads(decrypted_value)

        return ecr
