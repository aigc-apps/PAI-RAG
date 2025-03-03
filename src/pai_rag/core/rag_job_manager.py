import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from pai_rag.core.rag_knowledgebase_manager import (
    DEFAULT_KNOWLEDGE_PATH,
    DEFAULT_TASK_FILE,
)
from loguru import logger
import threading


class TaskInfo(BaseModel):
    task_id: str
    index_name: str
    input_files: List[str]
    last_modified_time: str
    status: str


class StageStatus(BaseModel):
    parse: Optional[str] = Field(None, description="Parse 阶段的状态")
    split: Optional[str] = Field(None, description="Split 阶段的状态")
    embed: Optional[str] = Field(None, description="Embed 阶段的状态")


class FileProcessInfo(BaseModel):
    status: StageStatus
    detail: Optional[str] = Field(None, description="任务详情")
    last_modified_time: str


class UploadJobManager:
    def __init__(
        self, task_file=DEFAULT_TASK_FILE, knowledge_path=DEFAULT_KNOWLEDGE_PATH
    ):
        self.task_file = task_file
        self.knowledge_path = knowledge_path
        self.lock = threading.Lock()  # 添加线程锁

    def _load_tasks(self):
        with self.lock:  # 确保线程安全
            if os.path.exists(self.task_file):
                with open(self.task_file, "r") as json_file:
                    try:
                        return json.load(json_file)
                    except json.JSONDecodeError:
                        return {}
            return {}

    def _save_tasks(self, tasks_dict):
        with self.lock:  # 确保线程安全
            with open(self.task_file, "w") as json_file:
                json.dump(tasks_dict, json_file, indent=4)

    def add_task_record(self, new_task_info):
        tasks_dict = self._load_tasks()
        tasks_dict[new_task_info["task_id"]] = new_task_info
        self._save_tasks(tasks_dict)

    def get_task_record(self, task_id):
        tasks_dict = self._load_tasks()
        if task_id in tasks_dict:
            return TaskInfo(**tasks_dict[task_id])
        else:
            raise ValueError(f"Task ID {task_id} not found.")

    def update_task_status(self, task_id, status):
        tasks_dict = self._load_tasks()
        if task_id in tasks_dict:
            tasks_dict[task_id]["status"] = status
            self._save_tasks(tasks_dict)
        else:
            raise ValueError(f"Task ID {task_id} not found.")

    def get_task_status(self, task_id):
        tasks_dict = self._load_tasks()
        if task_id in tasks_dict:
            return tasks_dict[task_id]["status"]
        else:
            return "not found"

    def track_job(
        self,
        task_id,
        index_name=None,
        input_files=None,
        stage=None,
        status=None,
        detail=None,
        start: bool = False,
        end: bool = False,
    ):
        index_jobs_folder = os.path.join(self.knowledge_path, index_name, ".logs")
        if start:
            _task_info = TaskInfo(
                task_id=task_id,
                index_name=index_name,
                input_files=input_files,
                last_modified_time=datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                status="start",
            )
            self.add_task_record(_task_info.model_dump())
        elif end:
            self.update_task_status(task_id, status="completed")
        elif status:
            self.update_task_status(task_id, status=status)
        else:
            logger.warning("No action specified for track_job.")

        single_task_record = self.get_task_record(task_id)
        single_task_details_file = os.path.join(
            index_jobs_folder, f"{single_task_record.last_modified_time}.json"
        )

        if not os.path.exists(single_task_details_file):
            single_task_info = self._create_single_task_info(
                stage, status, detail, single_task_record, single_task_details_file
            )
        else:
            with open(single_task_details_file, "r", encoding="utf-8") as json_file:
                single_task_info = json.load(json_file)
            self._update_single_task_info(
                single_task_info,
                stage,
                status,
                detail,
                single_task_record,
                single_task_details_file,
            )

    def _create_single_task_info(
        self, stage, status, detail, single_task_record, single_task_details_file
    ):
        task_details_info = {}

        input_files = single_task_record.input_files
        last_modified_time = single_task_record.last_modified_time
        if isinstance(input_files, list):
            for file in input_files:
                task_details_info[file] = FileProcessInfo(
                    status=StageStatus(
                        parse=status if stage == "parse" else None,
                        split=None,
                        embed=None,
                    ),
                    detail=detail,
                    last_modified_time=last_modified_time,
                ).model_dump()
        elif isinstance(input_files, str):
            task_details_info[input_files] = FileProcessInfo(
                status=StageStatus(
                    parse=status if stage == "parse" else None,
                    split=None,
                    embed=None,
                ),
                detail=detail,
                last_modified_time=last_modified_time,
            ).model_dump()
        else:
            raise ValueError("input_files must be a list or a string")
        self._write_single_task_info(task_details_info, single_task_details_file)

    def _update_single_task_info(
        self,
        jobs_info,
        stage,
        status,
        detail,
        single_task_record,
        single_task_details_file,
    ):
        input_files = single_task_record.input_files
        if isinstance(input_files, list):
            for file in input_files:
                if stage in jobs_info[file]["status"]:
                    jobs_info[file]["status"][stage] = status
                    jobs_info[file]["detail"] = detail
        elif isinstance(input_files, str):
            jobs_info[input_files]["status"][stage] = status
        self._write_single_task_info(jobs_info, single_task_details_file)

    def _write_single_task_info(self, jobs_info, jobs_info_file):
        try:
            with open(jobs_info_file, "w", encoding="utf-8") as json_file:
                json.dump(jobs_info, json_file, ensure_ascii=False, indent=4)
            logger.info(f"Successfully wrote data to {jobs_info_file}")
        except IOError as e:
            logger.error(f"Error writing to file {jobs_info_file}: {e}")


upload_job_manager = UploadJobManager()
