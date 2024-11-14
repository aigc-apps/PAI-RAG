import ray
import os
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
    # ray.init(runtime_env=RuntimeEnv(conda="pai_rag", working_dir="/home/ray/PAI-RAG"))
    ray.init(runtime_env={"working_dir": "/home/ray/PAI-RAG/", "conda": "pai_rag"})


@ray.remote
def write_to_file(results, filename):
    # 每个结果写入一个新行
    with open(filename, "a", encoding="utf-8") as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)  # 转换为 JSON 格式的字符串
            f.write(json_line + "\n")  # 每个 JSON 对象写入一行


@ray.remote
def process(input_file, args):
    from pai_rag.tools.data_process.tasks.load_and_parse_doc_task import (
        LoadAndParseDocTask,
    )

    task = LoadAndParseDocTask(**vars(args))
    res = task.process(input_file)
    return res


def main(args):
    init_ray_env()
    output = "/home/ray/PAI-RAG/localdata/test_LoadAndParseDocTask_ray.jsonl"
    dataset = get_dataset(args.data_path)
    run_tasks = [process.remote(data, args) for data in dataset]
    results = ray.get(run_tasks)
    print("Master node completed processing files.")
    write_to_file.remote(results, output)
    print(f"Results written to {output}' asynchronously.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--oss_path", type=str, default=None)
    args = parser.parse_args()

    print(f"Init: args: {args}")

    main(args)
