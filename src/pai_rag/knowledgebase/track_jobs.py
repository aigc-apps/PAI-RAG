import os
import json
from datetime import datetime
from pai_rag.knowledgebase.constants import DEFAULT_KNOWLEDGE_PATH, DEFAULT_TASK_FILE


def add_job_records(new_task_info):
    if os.path.exists(DEFAULT_TASK_FILE):
        with open(DEFAULT_TASK_FILE, "r") as json_file:
            try:
                tasks_dict = json.load(json_file)
            except json.JSONDecodeError:
                tasks_dict = {}
    else:
        tasks_dict = {}

    task_id = new_task_info["task_id"]
    tasks_dict[task_id] = new_task_info

    with open(DEFAULT_TASK_FILE, "w") as json_file:
        json.dump(tasks_dict, json_file, indent=4)


def get_job_records(task_id):
    if os.path.exists(DEFAULT_TASK_FILE):
        with open(DEFAULT_TASK_FILE, "r") as json_file:
            try:
                tasks_dict = json.load(json_file)
            except json.JSONDecodeError:
                tasks_dict = {}
    else:
        tasks_dict = {}
    if task_id in tasks_dict:
        return tasks_dict[task_id]
    else:
        raise ValueError(f"Task ID {task_id} not found.")


def track_jobs(
    task_id, index_name=None, input_files=None, stage=None, status=None, detail=None
):
    index_jobs_folder = os.path.join(DEFAULT_KNOWLEDGE_PATH, index_name, ".logs")
    os.makedirs(index_jobs_folder, exist_ok=True)
    if status == "start":
        _task_info = {
            "task_id": task_id,
            "index_name": index_name,
            "input_files": input_files,
            "last_modified_time": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        }
        add_job_records(_task_info)
    job_record = get_job_records(task_id)
    jobs_info_file = os.path.join(
        index_jobs_folder, f"{job_record['last_modified_time']}.json"
    )
    if not os.path.exists(jobs_info_file):
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
        try:
            with open(jobs_info_file, "w", encoding="utf-8") as json_file:
                json.dump(jobs_info, json_file, ensure_ascii=False, indent=4)
            print(f"成功将数据写入 {jobs_info_file}")
        except IOError as e:
            print(f"写入文件{jobs_info_file}时出错: {e}")
    else:
        with open(jobs_info_file, "r", encoding="utf-8") as json_file:
            jobs_info = json.load(json_file)
        if isinstance(input_files, list):
            for file in input_files:
                jobs_info[file]["status"][stage] = status
                jobs_info[file]["detail"] = detail
        elif isinstance(input_files, str):
            jobs_info[file]["status"][stage] = status
        try:
            with open(jobs_info_file, "w", encoding="utf-8") as json_file:
                json.dump(jobs_info, json_file, ensure_ascii=False, indent=4)
            print(f"成功将数据写入 {jobs_info_file}")
        except IOError as e:
            print(f"写入文件{jobs_info_file}时出错: {e}")
