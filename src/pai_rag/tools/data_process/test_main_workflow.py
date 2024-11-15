import argparse
import ray
from ray.data.datasource.filename_provider import _DefaultFilenameProvider


def split_node_task(row):
    return row


def main(args):
    ds = ray.data.read_json(
        "/home/xiaowen/xiaowen/github_code/PAI-RAG/localdata/test_LoadAndParseDocTask.jsonl"
    )

    ds = ds.flat_map(
        split_node_task, fn_kwargs={"config_file": args.config_file}, concurrency=2
    )

    ds.write_json(
        "/home/xiaowen/xiaowen/github_code/PAI-RAG/localdata/output",
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
