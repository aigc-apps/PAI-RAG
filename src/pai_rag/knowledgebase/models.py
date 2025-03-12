import time
from pydantic import BaseModel, Field
from typing import Dict
from enum import Enum

from pai_rag.utils.time_utils import get_current_time_str


class FileOperationType(int, Enum):
    ADD = 1
    UPDATE = 2
    DELETE = 3


class FileChange(BaseModel):
    task_id: str
    file_name: str
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
