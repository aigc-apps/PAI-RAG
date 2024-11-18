import argparse
import ray
from ray.data.datasource.filename_provider import _DefaultFilenameProvider
from pai_rag.tools.data_process.tasks.split_node_task import split_node_task
import os

os.environ["RAY_DEDUP_LOGS"] = "0"


def init_ray_env(working_dir):
    ray.init(runtime_env={"working_dir": working_dir})


def main(args):
    init_ray_env(args.working_dir)
    ds = ray.data.read_json(args.data_path)

    ds = ds.flat_map(split_node_task, fn_kwargs={"config_file": args.config_file})

    ds.write_json(
        args.output_path,
        filename_provider=_DefaultFilenameProvider(file_format="jsonl"),
        force_ascii=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--working_dir", type=str, default=None)
    args = parser.parse_args()

    print(f"Init: args: {args}")

    main(args)
