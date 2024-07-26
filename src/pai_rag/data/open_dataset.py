from abc import ABC, abstractmethod
from collections import defaultdict
import os
import gzip
import json
import urllib.request
import tarfile
from datasets import load_dataset

DEFAULT_DATASET_DIR = "datasets"


class OpenDataSet(ABC):
    @abstractmethod
    def load_qrels(self, type: str):
        """加载评测文件
        :param type: 要加载的数据集的类型 [train, dev, test]
        """
        pass

    @abstractmethod
    def load_related_corpus(self):
        """加载语料库相关文件"""
        pass


class MiraclOpenDataSet(OpenDataSet):
    def __init__(
        self, dataset_path: str = None, corpus_path: str = None, lang: str = "zh"
    ):
        self.dataset_path = dataset_path or os.path.join(DEFAULT_DATASET_DIR, "miracl")
        self.corpus_path = corpus_path or os.path.join(
            DEFAULT_DATASET_DIR, "miracl-corpus"
        )
        self.lang = lang
        if not os.path.exists(self.dataset_path):
            dataset_url = "https://pai-rag.oss-cn-hangzhou.aliyuncs.com/huggingface/datasets/miracl.tar.gz"
            file_path = os.path.join(DEFAULT_DATASET_DIR, "miracl.tar.gz")
            self._extract_and_download_dataset(
                dataset_url, file_path, self.dataset_path
            )
        else:
            print(
                f"[MiraclOpenDataSet] Dataset file already exists at {self.dataset_path}."
            )
        if not os.path.exists(self.corpus_path):
            dataset_url = "https://pai-rag.oss-cn-hangzhou.aliyuncs.com/huggingface/datasets/miracl-corpus.tar.gz"
            file_path = os.path.join(DEFAULT_DATASET_DIR, "miracl-corpus.tar.gz")
            self._extract_and_download_dataset(dataset_url, file_path, self.corpus_path)
        else:
            print(
                f"[MiraclOpenDataSet] Corpus file already exists at {self.corpus_path}."
            )

    def _extract_and_download_dataset(self, url, file_path, extract_path):
        file_path_dir = os.path.dirname(file_path)
        if not os.path.exists(file_path_dir):
            os.makedirs(file_path_dir)
        with urllib.request.urlopen(url) as response, open(file_path, "wb") as out_file:
            print(f"[MiraclOpenDataSet] Start downloading file {file_path} from {url}.")
            data = response.read()
            out_file.write(data)
            print("[MiraclOpenDataSet] Finish downloading.")
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        with tarfile.open(file_path, "r:gz") as tar:
            print(
                f"[MiraclOpenDataSet] Start extracting file {file_path} to {extract_path}."
            )
            tar.extractall(path=extract_path)
            print("[MiraclOpenDataSet] Finish extracting.")

    def load_qrels(self, type: str):
        file = f"{self.dataset_path}/miracl/miracl-v1.0-{self.lang}/qrels/qrels.miracl-v1.0-{self.lang}-{type}.tsv"
        print(
            f"[MiraclOpenDataSet] Loading qrels for MiraclDataSet with type {type} from {file}..."
        )
        qrels = defaultdict(dict)
        docids = set()
        with open(file, encoding="utf-8") as f:
            for line in f:
                qid, _, docid, rel = line.strip().split("\t")
                qrels[qid][docid] = int(rel)
                docids.add(docid)
        print(
            f"[MiraclOpenDataSet] Loaded qrels {len(qrels)}, docids {len(docids)} with type {type}"
        )
        return qrels, docids

    def load_topic(self, type: str):
        file = f"{self.dataset_path}/miracl/miracl-v1.0-{self.lang}/topics/topics.miracl-v1.0-{self.lang}-{type}.tsv"
        print(
            f"[MiraclOpenDataSet] Loading topic for MiraclDataSet with type {type} from {file}..."
        )
        qid2topic = {}
        with open(file, encoding="utf-8") as f:
            for line in f:
                qid, topic = line.strip().split("\t")
                qid2topic[qid] = topic
        print(f"[MiraclOpenDataSet] Loaded qid2topic {len(qid2topic)} with type {type}")
        return qid2topic

    def load_related_corpus(self):
        corpus_dir = f"{self.corpus_path}/miracl-corpus/miracl-corpus-v1.0-zh"
        docid2doc = {}

        nodes = set()
        for dirpath, _, filenames in os.walk(corpus_dir):
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
                    f"[MiraclOpenDataSet] Loaded nodes {len(nodes)} from file_path {file_path}"
                )

        print(f"[MiraclOpenDataSet] Loaded all nodes {len(nodes)}")
        return nodes, docid2doc

    def load_related_corpus_for_type(self, type: str):
        corpus_dir = f"{self.corpus_path}/miracl-corpus/miracl-corpus-v1.0-zh"
        _, docids = self.load_qrels(type=type)
        docid2doc = {}

        nodes = set()
        for dirpath, _, filenames in os.walk(corpus_dir):
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
                print(
                    f"[MiraclOpenDataSet] Loaded nodes {len(nodes)} with type {type} from file_path {file_path}"
                )

        print(f"[MiraclOpenDataSet] Loaded all nodes {len(nodes)} with type {type}")
        return nodes, docid2doc


class DuRetrievalDataSet(OpenDataSet):
    def __init__(self, dataset_path: str = None, corpus_path: str = None):
        self.dataset_path = dataset_path or os.path.join(
            DEFAULT_DATASET_DIR, "DuRetrieval-qrels"
        )
        self.corpus_path = corpus_path or os.path.join(
            DEFAULT_DATASET_DIR, "DuRetrieval"
        )
        if not os.path.exists(self.dataset_path):
            dataset_url = "https://pai-rag.oss-cn-hangzhou.aliyuncs.com/huggingface/datasets/DuRetrieval-qrels.tar.gz"
            file_path = os.path.join(DEFAULT_DATASET_DIR, "DuRetrieval-qrels.tar.gz")
            self._extract_and_download_dataset(
                dataset_url, file_path, self.dataset_path
            )
        else:
            print(
                f"[DuRetrievalDataSet] Dataset file already exists at {self.dataset_path}."
            )
        if not os.path.exists(self.corpus_path):
            dataset_url = "https://pai-rag.oss-cn-hangzhou.aliyuncs.com/huggingface/datasets/DuRetrieval.tar.gz"
            file_path = os.path.join(DEFAULT_DATASET_DIR, "DuRetrieval.tar.gz")
            self._extract_and_download_dataset(dataset_url, file_path, self.corpus_path)
        else:
            print(
                f"[DuRetrievalDataSet] Corpus file already exists at {self.corpus_path}."
            )

    def _extract_and_download_dataset(self, url, file_path, extract_path):
        file_path_dir = os.path.dirname(file_path)
        if not os.path.exists(file_path_dir):
            os.makedirs(file_path_dir)
        with urllib.request.urlopen(url) as response, open(file_path, "wb") as out_file:
            print(
                f"[DuRetrievalDataSet] Start downloading file {file_path} from {url}."
            )
            data = response.read()
            out_file.write(data)
            print("[DuRetrievalDataSet] Finish downloading.")
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        with tarfile.open(file_path, "r:gz") as tar:
            print(
                f"[DuRetrievalDataSet] Start extracting file {file_path} to {extract_path}."
            )
            tar.extractall(path=extract_path)
            print("[DuRetrievalDataSet] Finish extracting.")

    def load_qrels(self, type: str = "dev"):
        print(
            f"[DuRetrievalDataSet] Loading qrels for DuRetrievalDataSet with type {type} from {self.dataset_path}..."
        )
        qrels_path = f"{self.dataset_path}/DuRetrieval-qrels"
        qrels_ori = load_dataset(qrels_path)
        qrels = defaultdict(dict)
        for sample in qrels_ori[type]:
            qid = sample["qid"]
            docid = sample["pid"]
            rel = sample["score"]
            qrels[qid][docid] = int(rel)
        print(f"[DuRetrievalDataSet] Loaded qrels {len(qrels)} with type {type}")
        return qrels

    def load_related_corpus(self):
        corpus_path = f"{self.corpus_path}/DuRetrieval"
        docid2doc = {}
        qid2query = {}
        nodes = set()
        du_dataset = load_dataset(corpus_path)
        for sample in du_dataset["corpus"]:
            nodes.add(
                (
                    sample["id"],
                    sample["text"],
                    self.corpus_path,
                )
            )
            docid2doc[sample["id"]] = sample["text"]
        print(
            f"[DuRetrievalDataSet] Loaded nodes {len(nodes)} from file_path {self.corpus_path}"
        )

        for sample in du_dataset["queries"]:
            qid2query[sample["id"]] = sample["text"]
        print(
            f"[DuRetrievalDataSet] Loaded queries {len(nodes)} from file_path {self.corpus_path}"
        )

        print(f"[DuRetrievalDataSet] Loaded all nodes {len(nodes)}")
        return nodes, docid2doc, qid2query
