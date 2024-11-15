import argparse

# from pai_rag.tools.data_process.tasks.split_node_task import split_node_task
import ray
from ray.data.datasource.filename_provider import _DefaultFilenameProvider

# from .tasks.split_node_task import split_node_task


def init_ray_env():
    ray.init(runtime_env={"working_dir": "/PAI-RAG/", "conda": "pai_rag"})


def text_node_to_dict(node):
    from datetime import datetime

    return {
        "id": node.id_,
        "embedding": node.embedding,
        "metadata": {
            k: str(v) if isinstance(v, datetime) else v
            for k, v in node.metadata.items()
        },
        "text": node.text,
        "mimetype": node.mimetype,
        "start_char_idx": node.start_char_idx,
        "end_char_idx": node.end_char_idx,
        "text_template": node.text_template,
        "metadata_template": node.metadata_template,
        "metadata_seperator": node.metadata_seperator,
    }


def split_node_task(document, config_file):
    from pai_rag.core.rag_module import resolve
    from pai_rag.integrations.nodeparsers.pai.pai_node_parser import PaiNodeParser
    from llama_index.core.schema import Document

    from pai_rag.core.rag_config_manager import RagConfigManager

    config = RagConfigManager.from_file(config_file).get_value()
    parser_config = config.node_parser
    node_parser = resolve(cls=PaiNodeParser, parser_config=parser_config)
    format_doc = Document(
        doc_id=document["doc_id"],
        text=document["data"]["content"],
        metadata=document["data"]["meta_data"],
    )
    nodes = node_parser([format_doc])
    return [text_node_to_dict(node) for node in nodes]


def main(args):
    init_ray_env()
    ds = ray.data.read_json("/mnt/dlc/localdata/test_LoadAndParseDocTask_ray.jsonl")

    ds = ds.flat_map(
        split_node_task, fn_kwargs={"config_file": args.config_file}, concurrency=2
    )

    ds.write_json(
        "/mnt/dlc/localdata/outout_ray_split",
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
