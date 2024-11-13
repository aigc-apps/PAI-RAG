import argparse
import json
import os
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


def process(input_file, args):
    from pai_rag.tools.data_process.tasks.load_and_parse_doc_task import (
        LoadAndParseDocTask,
    )

    task = LoadAndParseDocTask(**vars(args))
    return task.process(input_file)


def main(args):
    dataset = get_dataset(args.data_path)
    run_tasks = [process(d, args) for d in dataset]

    results = run_tasks
    with open(
        "./localdata/test_LoadAndParseDocTask.jsonl", "w", encoding="utf-8"
    ) as file:
        for item in results:
            json_line = json.dumps(item, ensure_ascii=False)  # 将字典转换为 JSON 字符串
            file.write(json_line + "\n")  # 写入文件并加上换行符
    print("Master node completed processing and saved states.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--oss_path", type=str, default=None)
    args = parser.parse_args()

    print(f"Init: args: {args}")

    main(args)
