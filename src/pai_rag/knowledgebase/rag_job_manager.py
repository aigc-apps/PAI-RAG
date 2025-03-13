import asyncio
import traceback
import os
import json
import time
from typing import Dict, List, OrderedDict, Tuple
from pai_rag.core.models.errors import UserInputError
from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_module import resolve_task_executor
from pai_rag.knowledgebase.rag_knowledgebase import knowledgebase_manager
from pai_rag.knowledgebase.models import (
    FileChange,
    FileItem,
    FileOperationType,
    FileProcessResult,
    FileProcessStatus,
    JobStatus,
    TaskInfo,
)
from pai_rag.utils.constants import (
    DEFAILT_MAX_FILE_TASK_COUNT,
    DEFAULT_KNOWLEDGEBASE_PATH,
    DEFAULT_TASK_FILE,
)
from loguru import logger
import threading
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED

from pai_rag.utils.time_utils import get_current_time_str


DEFAULT_BACKGROUND_WORKER_NUM = os.environ.get("DEFAULT_BACKGROUND_WORKER_NUM", 4)

"""
文件信号短期容易出现重复提交，所以需要做防抖处理。
设置时间窗口为5s, 5s内重复提交的信号会合并任务队列。
5s内不被更新的信号会被当作稳定的信号取出。
"""


class TimeDebouncedTaskQueue:
    def __init__(self, max_size: int = 5000, time_window: int = 5):
        self.max_size = max_size
        self.time_window = time_window
        self.task_queue: OrderedDict[Tuple[str, str, int], FileItem] = {}
        self.lock = threading.Lock()

    def first(self):
        """Return the first element from an ordered collection
        or an arbitrary element from an unordered collection.
        Raise StopIteration if the collection is empty.
        """
        if not self.task_queue:
            return None

        return next(iter(self.task_queue))

    def _merge_old(self, item_key, item, cur_time):
        merged = False
        old_item = self.task_queue.get(item_key)
        if old_item is not None and old_item.timestamp + self.time_window > cur_time:
            logger.info(f"Merge file changes for {item_key}.")
            self.task_queue.pop(item_key)
            merged = True

        self.task_queue[item_key] = item
        return merged

    def _merge_with_two_key(self, item_key, item_key2, item, cur_time):
        merged = False
        old_item2 = self.task_queue.get(item_key2)
        if old_item2 is not None and old_item2.timestamp + self.time_window > cur_time:
            logger.info(f"Merge file changes for {item_key2}.")
            self.task_queue.pop(item_key2)
            merged = True

        old_item = self.task_queue.get(item_key)
        if old_item is not None and old_item.timestamp + self.time_window > cur_time:
            logger.info(f"Merge file changes for {item_key}.")
            self.task_queue.pop(item_key)
            merged = True

        self.task_queue[item_key] = item
        return merged

    def put(self, item: FileItem):
        with self.lock:
            cur_time = time.time()
            item_key = (item.knowledgebase, item.file_name, item.operation)
            if (
                item.operation == FileOperationType.DELETE
                or item.operation == FileOperationType.ADD
            ):
                merged = self._merge_old(item_key, item, cur_time)
            else:
                # update操作需要同时merge update和add
                item_key2 = (item.knowledgebase, item.file_name, FileOperationType.ADD)
                merged = self._merge_with_two_key(
                    item_key=item_key, item_key2=item_key2, item=item, cur_time=cur_time
                )
            return not merged

    def get(self):
        with self.lock:
            first_key = self.first()
            if first_key is not None:
                item = self.task_queue[first_key]
                if item.timestamp + self.time_window < time.time():
                    self.task_queue.pop(first_key)
                    return item
            return None


class JobManager:
    def __init__(
        self, task_file=DEFAULT_TASK_FILE, rag_config: RagConfig | None = None
    ):
        self.task_file = task_file
        self._lock = threading.Lock()
        self._job_status: JobStatus = self.load_status()
        self._task_queue = TimeDebouncedTaskQueue(max_size=DEFAILT_MAX_FILE_TASK_COUNT)
        self.rag_config = rag_config

    # 后续可以加resume机制
    def load_status(self):
        if not os.path.exists(self.task_file):
            return JobStatus()

        job_status = JobStatus.model_validate(json.load(open(self.task_file)))
        for knowledgebase in job_status.task_statuses:
            task_info = job_status.task_statuses[knowledgebase]
            for task in task_info.task_map:
                if (
                    task_info.task_map[task].status != FileProcessStatus.Done
                    or task_info.task_map[task].status != FileProcessStatus.Failed
                ):
                    task_info.task_map[task].status = FileProcessStatus.Failed
                    task_info.task_map[
                        task
                    ].failed_reason = "Task timeout. You can try reupload the files."
        return job_status

    def persist_task_status(self):
        with open(self.task_file, "w") as f:
            status_obj = self._job_status.model_dump()
            json.dump(status_obj, f, ensure_ascii=False)

    def update_config(self, new_config: RagConfig):
        self.rag_config = new_config

    def _get_task_executor(self, knowledgebase_name):
        knowledgebase = knowledgebase_manager.get_knowledgebase(knowledgebase_name)
        return resolve_task_executor(self.rag_config, knowledgebase)

    def _remove_file_prefix(self, knowledgebase_name: str, file_path: str):
        common_prefix = (
            "/" + DEFAULT_KNOWLEDGEBASE_PATH + f"/{knowledgebase_name}/docs/"
        )
        prefix_index = file_path.find(common_prefix)
        if prefix_index == -1:
            start_index = 0
        else:
            start_index = prefix_index + len(common_prefix)
        return file_path[start_index:]

    def get_job_history(self, name):
        if name not in self._job_status.task_statuses:
            return []

        task_history: Dict[str, FileItem] = self._job_status.task_statuses[
            name
        ].task_map
        return [
            {
                "task_id": task.task_id,
                "operation": task.operation.name,
                "file_name": self._remove_file_prefix(name, task.file_name),
                "status": task.status,
                "message": task.failed_reason,
                "last_modified_time": task.last_modified_time,
            }
            for _, task in task_history.items()
        ]

    def get_file_upload_status(self, knowledgebase_name: str, file_name: str):
        if knowledgebase_name not in self._job_status.task_statuses:
            raise UserInputError(f"knowledgebase {knowledgebase_name} not found.")

        if not file_name.startswith(DEFAULT_KNOWLEDGEBASE_PATH):
            file_name = os.path.join(
                DEFAULT_KNOWLEDGEBASE_PATH, knowledgebase_name, "docs", file_name
            )

        if file_name not in self._job_status.task_statuses[knowledgebase_name].task_map:
            raise UserInputError(
                f"File {file_name} not found in knowledgebase '{knowledgebase_name}'."
            )

        task = self._job_status.task_statuses[knowledgebase_name].task_map[file_name]

        return {
            "task_id": task.task_id,
            "operation": task.operation.name,
            "file_name": self._remove_file_prefix(knowledgebase_name, task.file_name),
            "status": task.status,
            "message": task.failed_reason,
            "last_modified_time": task.last_modified_time,
        }

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
                    file_item.file_name
                ] = file_item
            self.persist_task_status()

    def _update_task_status(
        self, file_item: FileItem, process_result: FileProcessResult
    ):
        self._job_status.task_statuses[file_item.knowledgebase].task_map[
            file_item.file_name
        ].status = process_result.status
        self._job_status.task_statuses[file_item.knowledgebase].task_map[
            file_item.file_name
        ].failed_reason = process_result.message
        self._job_status.task_statuses[file_item.knowledgebase].task_map[
            file_item.file_name
        ].last_modified_time = get_current_time_str()
        with self._lock:
            self.persist_task_status()

    def execute_job_with_workers(self, worker_num=DEFAULT_BACKGROUND_WORKER_NUM):
        try:
            asyncio.get_event_loop()
        except Exception as ex:
            logger.warning(f"No event loop found, will create new: {ex}")
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

        with ProcessPoolExecutor(max_workers=worker_num) as pool:
            running_tasks = []
            max_concurrent_task = worker_num * 3

            # 去重复，不让同一个文件同时处理（短时间上传多次同时处理会出现冲突）
            current_running_files = set()

            while True:
                try:
                    if len(running_tasks) >= max_concurrent_task:
                        logger.info(
                            f"Wait any '{len(running_tasks)}' tasks to complete before submitting."
                        )

                        completed_tasks, processing_tasks = wait(
                            running_tasks, return_when=FIRST_COMPLETED
                        )
                        running_tasks = list(processing_tasks)
                        for complete in completed_tasks:
                            item, result = complete.result()
                            current_running_files.remove(
                                (item.knowledgebase, item.file_name)
                            )
                            self._update_task_status(item, result)

                        logger.info(f"Now '{len(running_tasks)}' tasks running.")

                    if self.rag_config is None:
                        logger.debug("任务队列准备中...")
                        time.sleep(2)
                        continue
                    file_item: FileItem = self._task_queue.get()
                    if file_item is None:
                        # 队列空，清空所有运行中任务
                        if len(running_tasks) > 0:
                            logger.info(
                                f"No tasks dequeued. Wait all '{len(running_tasks)}' tasks to complete before submitting."
                            )
                            completed_tasks, processing_tasks = wait(
                                running_tasks, return_when=ALL_COMPLETED
                            )
                            running_tasks = list(processing_tasks)
                            for complete in completed_tasks:
                                item, result = complete.result()
                                current_running_files.remove(
                                    (item.knowledgebase, item.file_name)
                                )
                                self._update_task_status(item, result)

                        logger.debug("后台任务队列为空。sleeping...")
                        time.sleep(5)  # 后续还是要做成异步？
                        continue

                    file_key = (file_item.knowledgebase, file_item.file_name)
                    if file_key in current_running_files:
                        logger.info(
                            f"File {file_key} is already running. Wait all '{len(running_tasks)}' tasks to complete before submitting."
                        )

                        # 文件已经在执行中，清空所有运行任务再提交
                        if len(running_tasks) > 0:
                            completed_tasks, processing_tasks = wait(
                                running_tasks, return_when=ALL_COMPLETED
                            )
                        running_tasks = list(processing_tasks)
                        for complete in completed_tasks:
                            item, result = complete.result()
                            current_running_files.remove(
                                (item.knowledgebase, item.file_name)
                            )
                            self._update_task_status(item, result)
                        logger.info(f"{len(running_tasks)} tasks completed.")

                    current_running_files.add(file_key)

                    logger.info(
                        f"开始处理: TaskId:{file_item.task_id} 文件: {file_item.file_name} 知识库: {file_item.knowledgebase} operation{file_item.operation}."
                    )
                    task_executor = self._get_task_executor(file_item.knowledgebase)
                    new_task = pool.submit(task_executor.run_once, file_item)  # 不支持流式返回
                    self._job_status.task_statuses[file_item.knowledgebase].task_map[
                        file_item.file_name
                    ].status = FileProcessStatus.Processing

                    running_tasks.append(new_task)
                except Exception:
                    logger.error(f"后台任务队列处理出错: {traceback.format_exc()}")
                    pass

    def execute_job(self):
        try:
            asyncio.get_event_loop()
        except Exception as ex:
            logger.warning(f"No event loop found, will create new: {ex}")
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

        while True:
            if self.rag_config is None:
                logger.debug("任务队列准备中...")
                time.sleep(2)
                continue
            file_item: FileItem = self._task_queue.get()
            if file_item is None:
                logger.debug("后台任务队列为空。sleeping...")
                time.sleep(5)  # 后续还是要做成异步？
                continue

            try:
                logger.info(
                    f"开始处理: TaskId:{file_item.task_id} 文件: {file_item.file_name} 知识库: {file_item.knowledgebase} operation{file_item.operation}."
                )
                executor = self._get_task_executor(file_item.knowledgebase)
                for resp in executor.run(file_item):
                    self._job_status.task_statuses[file_item.knowledgebase].task_map[
                        file_item.file_name
                    ].status = resp.status
                    self._job_status.task_statuses[file_item.knowledgebase].task_map[
                        file_item.file_name
                    ].failed_reason = resp.message
                    self._job_status.task_statuses[file_item.knowledgebase].task_map[
                        file_item.file_name
                    ].last_modified_time = get_current_time_str()

                if (
                    self._job_status.task_statuses[file_item.knowledgebase]
                    .task_map[file_item.file_name]
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
                            .task_map[file_item.file_name]
                            .last_modified_time,
                        )

                logger.info(
                    f"处理完成: TaskId:{file_item.task_id} 文件: {file_item.file_name} 知识库:{file_item.knowledgebase}."
                )
                with self._lock:
                    self.persist_task_status()

            except Exception as ex:
                logger.error(
                    f"后台任务队列处理 '{file_item.file_name}' 出错: {traceback.format_exc()}"
                )
                self._job_status.task_statuses[file_item.knowledgebase].task_map[
                    file_item.file_name
                ].status = FileProcessStatus.Failed
                self._job_status.task_statuses[file_item.knowledgebase].task_map[
                    file_item.file_name
                ].failed_reason = str(ex)
                self._job_status.task_statuses[file_item.knowledgebase].task_map[
                    file_item.file_name
                ].last_modified_time = get_current_time_str()
                with self._lock:
                    self.persist_task_status()


job_manager = JobManager()
