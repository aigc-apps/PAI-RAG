from collections import defaultdict
import os
import gzip
import json

DEFAULT_TOPIC_DEV_FILE = (
    "local_path/miracl/miracl-v1.0-zh/topics/topics.miracl-v1.0-zh-dev.tsv"
)
DEFAULT_QRELS_DEV_FILE = (
    "local_path/miracl/miracl-v1.0-zh/qrels/qrels.miracl-v1.0-zh-dev.tsv"
)
DEFAULT_CORPUS_DIR = "local_path/miracl-corpus/miracl-corpus-v1.0-zh"


def load_topic(fn: str = None):
    if fn is None:
        fn = DEFAULT_TOPIC_DEV_FILE
    qid2topic = {}
    with open(fn, encoding="utf-8") as f:
        for line in f:
            qid, topic = line.strip().split("\t")
            qid2topic[qid] = topic
    print(f"[Miracl Dataset] Loaded qid2topic {len(qid2topic)}")
    return qid2topic


def load_qrels(fn: str = None):
    if fn is None:
        fn = DEFAULT_QRELS_DEV_FILE

    qrels = defaultdict(dict)
    docids = set()
    with open(fn, encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split("\t")
            qrels[qid][docid] = int(rel)
            docids.add(docid)
    print(f"[Miracl Dataset] Loaded qrels {len(qrels)}, docids {len(docids)}")
    return qrels, docids


def load_related_docs_for_dev(dir: str = None):
    if dir is None:
        dir = DEFAULT_CORPUS_DIR

    _, docids = load_qrels(DEFAULT_QRELS_DEV_FILE)
    docid2doc = {}

    nodes = set()
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                for line in f:
                    json_obj = json.loads(line)
                    if json_obj["docid"] in docids:
                        nodes.add(
                            (
                                json_obj["docid"],
                                json_obj["text"],
                                json_obj["title"],
                                file_path,
                            )
                        )
                        docid2doc[json_obj["docid"]] = json_obj["text"]

    print(f"[Miracl Dataset] Loaded dev nodes {len(nodes)}")
    return nodes, docid2doc


def load_related_docs_all(dir: str = None):
    if dir is None:
        dir = DEFAULT_CORPUS_DIR

    docid2doc = {}

    nodes = set()
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                for line in f:
                    json_obj = json.loads(line)
                    nodes.add(
                        (
                            json_obj["docid"],
                            json_obj["text"],
                            json_obj["title"],
                            file_path,
                        )
                    )
                    docid2doc[json_obj["docid"]] = json_obj["text"]
            print(
                f"[Miracl Dataset] Loaded nodes {len(nodes)} for file_path {file_path}"
            )

    print(f"[Miracl Dataset] Loaded all nodes {len(nodes)}")
    return nodes, docid2doc
