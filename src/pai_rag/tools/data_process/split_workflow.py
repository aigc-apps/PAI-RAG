import argparse
from pai_rag.tools.data_process.tasks.split_node_task import split_node_task
import ray
from ray.data.datasource.filename_provider import _DefaultFilenameProvider


def init_ray_env():
    ray.init(runtime_env={"working_dir": "/home/ray/PAI-RAG/", "conda": "pai_rag"})


def main(args):
    init_ray_env()
    ds = ray.data.read_json(
        "/home/ray/PAI-RAG/localdata/test_LoadAndParseDocTask_ray.jsonl"
    )

    ds = ds.flat_map(
        split_node_task, fn_kwargs={"config_file": args.config_file}, concurrency=2
    )

    ds.write_json(
        "/home/ray/PAI-RAG/localdata/outout_ray_split",
        filename_provider=_DefaultFilenameProvider(file_format="jsonl"),
        force_ascii=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--oss_path", type=str, default=None)
    args = parser.parse_args()

    print(f"Init: args: {args}")

    main(args)
