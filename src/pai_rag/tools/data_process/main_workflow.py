import ray
import os
from ray.runtime_env import RuntimeEnv
import argparse
import json
from typing import List


def get_dataset(
    file_path_or_directory: str | List[str],
    filter_pattern: str = None,
):
    filter_pattern = filter_pattern or "*"

    if isinstance(file_path_or_directory, list):
        # file list
        input_files = [f for f in file_path_or_directory if os.path.isfile(f)]
    elif isinstance(file_path_or_directory, str) and os.path.isdir(
        file_path_or_directory
    ):
        # glob from directory
        import pathlib

        directory = pathlib.Path(file_path_or_directory)
        input_files = [f for f in directory.rglob(filter_pattern) if os.path.isfile(f)]
    else:
        # Single file
        input_files = [file_path_or_directory]

    if not input_files:
        raise ValueError(
            f"No file found at path '{file_path_or_directory}' with pattern '{filter_pattern}'."
        )
    return input_files


def init_ray_env():
    ray.init(runtime_env=RuntimeEnv(conda="pai_rag"))


@ray.remote
class LockActor:
    def __init__(self):
        self.lock = ray.util.lock.Lock()  # 创建分布式锁

    def acquire(self):
        self.lock.acquire()  # 获取锁

    def release(self):
        self.lock.release()  # 释放锁

    def write_to_file(self, data, filename):
        with open(filename, "w", encoding="utf-8") as file:
            json_line = json.dumps(data, ensure_ascii=False)  # 将字典转换为 JSON 字符串
            file.write(json_line + "\n")  # 写入文件并加上换行符


@ray.remote
def process(lock_actor, input_file, output, args):
    from pai_rag.tools.data_process.tasks.load_and_parse_doc_task import (
        LoadAndParseDocTask,
    )

    task = LoadAndParseDocTask(**vars(args))
    res = task.process(input_file)

    lock_actor.acquire.remote()  # 获取锁
    try:
        lock_actor.write_to_file.remote(res, output)  # 写入文件
    finally:
        lock_actor.release.remote()  # 确保释放锁
    return {"file": input_file, "status": "success"}


def main(args):
    init_ray_env()
    lock_actor = LockActor.remote()
    output = "/app/localdata/test_LoadAndParseDocTask_ray.jsonl"
    dataset = get_dataset(args.data_path)
    run_tasks = [process.remote(lock_actor, data, output, args) for data in dataset]
    results = ray.get(run_tasks)
    with open(
        "/app/localdata/test_LoadAndParseDocTask_status_ray.json", "w"
    ) as status_file:
        json.dump(results, status_file, indent=4)
    print("Master node completed processing and saved states.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--oss_path", type=str, default=None)
    args = parser.parse_args()

    print(f"Init: args: {args}")

    main(args)
