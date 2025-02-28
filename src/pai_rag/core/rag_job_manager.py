import os
import json
from datetime import datetime
from pai_rag.core.rag_knowledgebase_manager import (
    DEFAULT_KNOWLEDGE_PATH,
    DEFAULT_TASK_FILE,
)
from loguru import logger


class UploadJobManager:
    def __init__(
        self, task_file=DEFAULT_TASK_FILE, knowledge_path=DEFAULT_KNOWLEDGE_PATH
    ):
        self.task_file = task_file
        self.knowledge_path = knowledge_path

    def _load_tasks(self):
        if os.path.exists(self.task_file):
            with open(self.task_file, "r") as json_file:
                try:
                    return json.load(json_file)
                except json.JSONDecodeError:
                    return {}
        return {}

    def _save_tasks(self, tasks_dict):
        with open(self.task_file, "w") as json_file:
            json.dump(tasks_dict, json_file, indent=4)

    def add_job_record(self, new_task_info):
        tasks_dict = self._load_tasks()
        task_id = new_task_info["task_id"]
        tasks_dict[task_id] = new_task_info
        self._save_tasks(tasks_dict)

    def get_job_record(self, task_id):
        tasks_dict = self._load_tasks()
        if task_id in tasks_dict:
            return tasks_dict[task_id]
        else:
            raise ValueError(f"Task ID {task_id} not found.")

    def track_job(
        self,
        task_id,
        index_name=None,
        input_files=None,
        stage=None,
        status=None,
        detail=None,
    ):
        index_jobs_folder = os.path.join(self.knowledge_path, index_name, ".logs")
        os.makedirs(index_jobs_folder, exist_ok=True)

        if status == "start":
            _task_info = {
                "task_id": task_id,
                "index_name": index_name,
                "input_files": input_files,
                "last_modified_time": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            }
            self.add_job_record(_task_info)

        job_record = self.get_job_record(task_id)
        jobs_info_file = os.path.join(
            index_jobs_folder, f"{job_record['last_modified_time']}.json"
        )

        if not os.path.exists(jobs_info_file):
            jobs_info = self._create_jobs_info(
                input_files, stage, status, detail, job_record
            )
            self._write_jobs_info(jobs_info, jobs_info_file)
        else:
            with open(jobs_info_file, "r", encoding="utf-8") as json_file:
                jobs_info = json.load(json_file)
            self._update_jobs_info(jobs_info, input_files, stage, status, detail)
            self._write_jobs_info(jobs_info, jobs_info_file)

    def _create_jobs_info(self, input_files, stage, status, detail, job_record):
        jobs_info = {}
        if isinstance(input_files, list):
            for file in input_files:
                jobs_info[file] = {
                    "status": {
                        "parse": status if stage == "parse" else None,
                        "split": None,
                        "embed": None,
                    },
                    "detail": detail,
                    "last_modified_time": job_record["last_modified_time"],
                }
        elif isinstance(input_files, str):
            jobs_info[input_files] = {
                "status": {
                    "parse": status if stage == "parse" else None,
                    "split": None,
                    "embed": None,
                },
                "detail": detail,
                "last_modified_time": job_record["last_modified_time"],
            }
        else:
            raise ValueError("input_files must be a list or a string")
        return jobs_info

    def _update_jobs_info(self, jobs_info, input_files, stage, status, detail):
        if isinstance(input_files, list):
            for file in input_files:
                if stage in jobs_info[file]["status"]:
                    jobs_info[file]["status"][stage] = status
                    jobs_info[file]["detail"] = detail
        elif isinstance(input_files, str):
            jobs_info[input_files]["status"][stage] = status

    def _write_jobs_info(self, jobs_info, jobs_info_file):
        try:
            with open(jobs_info_file, "w", encoding="utf-8") as json_file:
                json.dump(jobs_info, json_file, ensure_ascii=False, indent=4)
            logger.info(f"Successfully wrote data to {jobs_info_file}")
        except IOError as e:
            logger.error(f"Error writing to file {jobs_info_file}: {e}")


upload_job_manager = UploadJobManager()
