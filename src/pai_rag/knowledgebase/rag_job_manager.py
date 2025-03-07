import asyncio
import traceback
import os
import json
from queue import Empty, Queue
import time
from typing import List
from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_module import resolve_task_executor
from pai_rag.knowledgebase.rag_knowledgebase import knowledgebase_manager
from pai_rag.knowledgebase.models import (
    FileChange,
    FileItem,
    FileOperationType,
    FileProcessStatus,
    JobStatus,
    TaskInfo,
)
from pai_rag.utils.constants import (
    DEFAILT_MAX_FILE_TASK_COUNT,
    DEFAULT_TASK_FILE,
)
from loguru import logger
import threading

from pai_rag.utils.time_utils import get_current_time_str


class JobManager:
    def __init__(
        self, task_file=DEFAULT_TASK_FILE, rag_config: RagConfig | None = None
    ):
        self.task_file = task_file
        self._lock = threading.Lock()
        self._job_status: JobStatus = self.load_status()
        self._task_queue = Queue(maxsize=DEFAILT_MAX_FILE_TASK_COUNT)
        self.rag_config = rag_config

    # 后续可以加resume机制
    def load_status(self):
        if not os.path.exists(self.task_file):
            return JobStatus()

        return JobStatus.model_validate(json.load(open(self.task_file)))

    def persist_task_status(self):
        with open(self.task_file, "w") as f:
            status_obj = self._job_status.model_dump()
            json.dump(status_obj, f, ensure_ascii=False)

    def update_config(self, new_config: RagConfig):
        self.rag_config = new_config

    def _get_task_executor(self, knowledgebase_name):
        knowledgebase = knowledgebase_manager.get_knowledgebase(knowledgebase_name)
        return resolve_task_executor(self.rag_config, knowledgebase)

    def get_job_history(self, name):
        if name not in self._job_status.task_statuses:
            return []

        task_history = self._job_status.task_statuses[name].task_map
        return [
            {
                "task_id": task_id,
                "file_name": task.file_name,
                "status": task.status,
                "message": task.failed_reason,
                "last_modified_time": task.last_modified_time,
            }
            for task_id, task in task_history.items()
        ]

    def submit_job(self, file_changes: List[FileChange]):
        with self._lock:
            for file_change in file_changes:
                if file_change.knowledgebase not in self._job_status.task_statuses:
                    self._job_status.task_statuses[
                        file_change.knowledgebase
                    ] = TaskInfo(
                        knowledgebase=file_change.knowledgebase, task_file_map={}
                    )

                file_item = FileItem(
                    task_id=file_change.task_id,
                    knowledgebase=file_change.knowledgebase,
                    file_name=file_change.file_name,
                    operation=file_change.operation,
                    status=FileProcessStatus.PENDING,
                )
                self._task_queue.put(file_item)
                self._job_status.task_statuses[file_change.knowledgebase].task_map[
                    file_change.task_id
                ] = file_item
            self.persist_task_status()

    def execute_job(self):
        while True:
            if self.rag_config is None:
                logger.debug("任务队列准备中...")
                time.sleep(2)
                continue
            try:
                file_item: FileItem = self._task_queue.get_nowait()
                logger.info(
                    f"开始处理: TaskId:{file_item.task_id} 文件: {file_item.file_name} 知识库: {file_item.knowledgebase} operation{file_item.operation}."
                )
                executor = self._get_task_executor(file_item.knowledgebase)
                for resp in executor.run(file_item):
                    self._job_status.task_statuses[file_item.knowledgebase].task_map[
                        file_item.task_id
                    ].status = resp.status
                    self._job_status.task_statuses[file_item.knowledgebase].task_map[
                        file_item.task_id
                    ].failed_reason = resp.message
                    self._job_status.task_statuses[file_item.knowledgebase].task_map[
                        file_item.task_id
                    ].last_modified_time = get_current_time_str()

                if (
                    self._job_status.task_statuses[file_item.knowledgebase]
                    .task_map[file_item.task_id]
                    .status
                    == FileProcessStatus.Done
                ):
                    if file_item.operation == FileOperationType.DELETE:
                        knowledgebase_manager.delete_doc_from_knowledgebase(
                            file_item.knowledgebase, file_item.file_name
                        )
                    else:
                        knowledgebase_manager.add_doc_to_knowledgebase(
                            knowledgebase_name=file_item.knowledgebase,
                            doc_id=file_item.task_id,
                            file_name=file_item.file_name,
                            last_modified_time=self._job_status.task_statuses[
                                file_item.knowledgebase
                            ]
                            .task_map[file_item.task_id]
                            .last_modified_time,
                        )

                logger.info(
                    f"处理完成: TaskId:{file_item.task_id} 文件: {file_item.file_name} 知识库:{file_item.knowledgebase}."
                )
                with self._lock:
                    self.persist_task_status()

            except Empty:
                logger.debug("后台任务队列为空。sleeping...")
                time.sleep(5)  # 后续还是要做成异步？
            except Exception:
                logger.error(
                    f"后台任务队列处理 '{file_item.file_name}' 出错: {traceback.format_exc()}"
                )

    async def execute_job_async(self):
        while True:
            if self.rag_config is None:
                logger.debug("任务队列准备中...")
                await asyncio.sleep(2)
                continue
            try:
                file_item: FileItem = self._task_queue.get_nowait()
                logger.info(
                    f"开始处理: TaskId:{file_item.task_id} 文件: {file_item.file_name} 知识库: {file_item.knowledgebase}."
                )
                executor = self._get_task_executor(file_item.knowledgebase)
                async for resp in executor.arun(file_item):
                    self._job_status.task_statuses[file_item.knowledgebase].task_map[
                        file_item.task_id
                    ].status = resp.status
                    self._job_status.task_statuses[file_item.knowledgebase].task_map[
                        file_item.task_id
                    ].failed_reason = resp.message
                    self._job_status.task_statuses[file_item.knowledgebase].task_map[
                        file_item.task_id
                    ].last_modified_time = get_current_time_str()

                if (
                    self._job_status.task_statuses[file_item.knowledgebase]
                    .task_map[file_item.task_id]
                    .status
                    == FileProcessStatus.Done
                ):
                    if file_item.operation == FileOperationType.DELETE:
                        knowledgebase_manager.delete_doc_from_knowledgebase(
                            file_item.knowledgebase, file_item.file_name
                        )
                    else:
                        knowledgebase_manager.add_doc_to_knowledgebase(
                            knowledgebase_name=file_item.knowledgebase,
                            doc_id=file_item.task_id,
                            file_name=file_item.file_name,
                            last_modified_time=self._job_status.task_statuses[
                                file_item.knowledgebase
                            ]
                            .task_map[file_item.task_id]
                            .last_modified_time,
                        )

                logger.info(
                    f"处理完成: TaskId:{file_item.task_id} 文件: {file_item.file_name} 知识库:{file_item.knowledgebase}."
                )

                with self._lock:
                    self.persist_task_status()

            except Empty:
                logger.debug("后台任务队列为空。sleeping...")

                await asyncio.sleep(5)  # 后续还是要做成异步？


job_manager = JobManager()
