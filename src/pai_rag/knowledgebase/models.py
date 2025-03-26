import time
from pydantic import BaseModel, Field, model_validator
from typing import Annotated, Dict, Union
from enum import Enum

from pai_rag.integrations.embeddings.pai.pai_embedding_config import PaiBaseEmbeddingConfig
from pai_rag.integrations.index.pai.vector_store_config import BaseVectorStoreConfig
from pai_rag.utils.constants import DEFAULT_KNOWLEDGEBASE_NAME
from pai_rag.utils.time_utils import get_current_time_str


class FileOperationType(int, Enum):
    ADD = 1
    UPDATE = 2
    DELETE = 3


class FileChange(BaseModel):
    task_id: str
    file_name: str
    file_hash: str
    operation: FileOperationType
    knowledgebase: str


class FileProcessStatus(str, Enum):
    PENDING = "pending"
    Parsing = "parsing"
    Chunking = "chunking"
    Embedding = "embedding"
    Persisting = "persisting"
    Done = "done"
    Failed = "failed"


class FileItem(FileChange):
    status: FileProcessStatus
    last_modified_time: str = Field(default_factory=lambda: get_current_time_str())
    timestamp: float = Field(default_factory=lambda: time.time())
    failed_reason: str | None = None


class FileProcessResult(BaseModel):
    status: FileProcessStatus
    message: str | None = None


class TaskInfo(BaseModel):
    knowledgebase: str
    task_map: Dict[str, FileItem] = {}
    last_modified_time: str = Field(default_factory=lambda: get_current_time_str())


class JobStatus(BaseModel):
    task_statuses: Dict[str, TaskInfo] = {}


class KnowledgeBase(BaseModel):
    name: str = Field(
        default=DEFAULT_KNOWLEDGEBASE_NAME,
        description="Knowledgebase name.",
        pattern=r"^[0-9a-zA-Z_-]{3, 20}$",
    )

    vector_store_config: Annotated[
        Union[BaseVectorStoreConfig.get_subclasses()], Field(discriminator="type")
    ]
    embedding_config: Annotated[
        Union[PaiBaseEmbeddingConfig.get_subclasses()], Field(discriminator="source")
    ]

    @model_validator(mode="before")
    def preprocess(cls, values: Dict) -> Dict:
        if "index_name" in values:
            values["name"] = values["index_name"]
        return values
