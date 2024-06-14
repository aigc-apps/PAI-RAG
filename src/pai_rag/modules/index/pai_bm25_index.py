import logging
import os
import pickle
import datetime
import json
import numpy as np
import pandas as pd
from typing import Callable, List, cast, Dict
from llama_index.core.schema import BaseNode, TextNode, NodeWithScore
from pai_rag.utils.tokenizer import jieba_tokenizer
import concurrent.futures
from scipy.sparse import csr_matrix


logger = logging.getLogger(__name__)

MAX_DOC_LIMIT = 9000000000  # 9B

"""
Store structure:
000001.json
000002.json
...
999999.json
"""
DEFAULT_STORE_DIR = "local_bm25_store"

# PART SIZE
DEFAULT_FILE_PART_DIR = "parts"
DEFAULT_PART_SIZE = 10000

"""
Index structure:
{
doc_count: 0,
doc_len_sum: 0,
inverted_index: []
}
"""
DEFAULT_INDEX_FILE = "bm25.index.pkl"
DEFAULT_INDEX_MATRIX_FILE = "bm25.index.matrix.pkl"


class LocalBm25Index:
    def __init__(self):
        self.doc_count: int = 0
        self.token_count: int = 0
        self.doc_lens: np.ndarray = np.array([])
        self.doc_token_index: List[Dict[str, int]] = []
        self.inverted_index: List[set[int]] = []
        self.token_map: Dict[str, int] = {}
        self.node_id_map: Dict[str, int] = {}


class PaiBm25Index:
    def __init__(
        self,
        persist_path: str,
        k1: int = 1.5,
        b: int = 0.75,
        tokenizer: Callable = None,
        workers: int = 5,
    ):
        self.k1 = k1
        self.b = b

        self.persist_path = os.path.join(persist_path, DEFAULT_STORE_DIR)
        self.parts_path = os.path.join(self.persist_path, DEFAULT_FILE_PART_DIR)
        self.index_file = os.path.join(self.persist_path, DEFAULT_INDEX_FILE)
        self.index_matrix_file = os.path.join(
            self.persist_path, DEFAULT_INDEX_MATRIX_FILE
        )

        self.workers = workers

        self.tokenizer = tokenizer or jieba_tokenizer

        logger.info("Start loading local BM25 index!")
        if os.path.exists(self.parts_path):
            with open(self.index_file, "rb") as f:
                self.index: LocalBm25Index = pickle.load(f)
            with open(self.index_matrix_file, "rb") as f:
                self.index_matrix = pickle.load(f)
        else:
            self.index = LocalBm25Index()
            self.index_matrix = None
            self.token_map = {}

        logger.info("Finished loading BM25 index!")

    def split_doc(self, text_list: List[str], tokenizer: Callable):
        tokens_list = []
        for text in text_list:
            tokens = tokenizer(text)
            tokens_list.append(tokens)

        logger.info(f"Finished {len(text_list)} docs.")
        return tokens_list

    def persist(self, doc_list):
        print(f"{datetime.datetime.now()} start persisting!")

        os.makedirs(self.parts_path, exist_ok=True)

        with open(self.index_file, "wb") as wf:
            pickle.dump(self.index, wf)

        with open(self.index_matrix_file, "wb") as wf:
            pickle.dump(self.index_matrix, wf)

        start_pos = 0
        part_i = 1

        logger.info(f"Write to index with {len(doc_list)} docs")
        while start_pos < len(doc_list):
            part_file_name = os.path.join(self.parts_path, f"{part_i:06}.part")
            # TODO write with bucket
            with open(part_file_name, "w") as wf:
                wf.write("\n".join(doc_list[start_pos : start_pos + DEFAULT_PART_SIZE]))
                start_pos += DEFAULT_PART_SIZE
                part_i += 1
        logger.info("write index succeeded!")

    def add_docs(self, nodes: List[BaseNode]):
        node_index_list = []
        text_list = []
        id_list = []
        metadata_list = []

        for node in nodes:
            if isinstance(node, TextNode):
                text_node = cast(TextNode, node)
                node_id = text_node.node_id
                id_list.append(node_id)
                metadata_list.append(text_node.metadata)
                text_list.append(text_node.get_content())

                if node_id not in self.index.node_id_map:
                    node_index = self.index.doc_count
                    self.index.doc_count += 1
                    self.index.node_id_map[node_id] = node_index
                node_index_list.append(self.index.node_id_map[node_id])

            else:
                # don't handle image or graph node
                pass

        pad_size = self.index.doc_count - len(self.index.doc_lens)
        self.index.doc_lens = np.lib.pad(
            self.index.doc_lens, (0, pad_size), "constant", constant_values=(0)
        )

        logger.info(f"Start splitting {len(text_list)}")
        chunk_size = 100000
        start_pos = 0
        if len(text_list) < 2 * chunk_size:
            tokens_list = self.split_doc(text_list, self.tokenizer)
            self.process_token_list(tokens_list, id_list)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers
            ) as executor:
                futures = []
                future2startpos = {}
                while start_pos < len(text_list):
                    fut = executor.submit(
                        self.split_doc,
                        text_list[start_pos : start_pos + chunk_size],
                        self.tokenizer,
                    )
                    futures.append(fut)
                    future2startpos[fut] = start_pos
                    start_pos += chunk_size

                i = 0
                for fut in concurrent.futures.as_completed(futures):
                    start_pos = future2startpos[fut]
                    i += 1
                    logger.info(f"Finished future {i}, {start_pos}")
                    tokens_list = fut.result()
                    batch_id_list = id_list[start_pos : start_pos + chunk_size]
                    self.process_token_list(tokens_list, batch_id_list)

        self.construct_index_matrix()

        doc_list = [
            json.dumps(
                {"id": id_list[i], "text": text_list[i], "metadata": metadata_list[i]},
                ensure_ascii=False,
            )
            for i in range(len(text_list))
        ]
        self.persist(doc_list)

        logger.info("Successfully write to BM25 index.")
        return

    def construct_index_matrix(self):
        logger.info("Constructing index matrix...")

        # m * n matrix
        m = self.index.doc_count
        n = self.index.token_count

        df = np.array([len(i) for i in self.index.inverted_index])
        idf = np.log(1 + (m - df + 0.5) / (df + 0.5))

        avg_dl = np.average(self.index.doc_lens)
        dl_factor = self.k1 * (1 - self.b + self.b * (self.index.doc_lens) / avg_dl)

        rows = []
        cols = []
        data = []
        for i, doc_token_set in enumerate(self.index.doc_token_index):
            for token in doc_token_set:
                rows.append(i)
                cols.append(token)
                tf = doc_token_set[token]
                v = idf[token] * tf * (self.k1 + 1) / (tf + dl_factor[i])
                data.append(v)

        self.index_matrix = csr_matrix((data, (rows, cols)), shape=(m, n))

    def load_batch_from_part_file(self, batch_ids, part_id):
        nodes = []
        part_file_name = os.path.join(self.parts_path, f"{part_id+1:06}.part")
        with open(part_file_name, "r") as part_file:
            lines = part_file.readlines()
            for index in batch_ids:
                index = index % DEFAULT_PART_SIZE
                raw_text = lines[index]
                json_data = json.loads(raw_text)
                nodes.append(
                    TextNode(
                        id_=json_data["id"],
                        text=json_data["text"],
                        metadata=json_data["metadata"],
                    )
                )
        return nodes

    def load_docs_with_index(self, doc_indexes):
        results = []
        if len(doc_indexes) == 0:
            return results

        index2nodes = {}
        node_indexes = doc_indexes.copy()
        node_indexes.sort()
        bucket_size = 10000
        batch_ids = []
        pre_bucket = -1
        for i in doc_indexes:
            bucket = int(i / bucket_size)
            if bucket != pre_bucket:
                if batch_ids:
                    batch_nodes = self.load_batch_from_part_file(batch_ids, bucket)
                    index2nodes.update(zip(batch_ids, batch_nodes))
                    batch_ids = []
                pre_bucket = bucket
            batch_ids.append(i)

        if batch_ids:
            batch_nodes = self.load_batch_from_part_file(batch_ids, bucket)
            index2nodes.update(zip(batch_ids, batch_nodes))

        return [index2nodes[i] for i in node_indexes]

    def query(self, query_str: str, top_n: int = 5) -> List[NodeWithScore]:
        results = []
        if self.index_matrix is None:
            return results

        tokens = self.tokenizer(query_str)
        query_vec = np.zeros(self.index.token_count)
        for token in tokens:
            if token in self.index.token_map:
                query_vec[self.index.token_map[token]] += 1

        doc_scores = self.index_matrix.multiply(query_vec).sum(axis=1).getA1()
        doc_indexes = doc_scores.argsort()[::-1][:top_n]
        text_nodes = self.load_docs_with_index(doc_indexes)
        for i, node in enumerate(text_nodes):
            results.append(NodeWithScore(node=node, score=doc_scores[doc_indexes[i]]))

        return results

    def process_token_list(self, tokens_list, ids_list):
        for i, tokens in enumerate(tokens_list):
            token_index_set = {}
            for token in tokens:
                if token not in self.index.token_map:
                    self.index.token_map[token] = self.index.token_count
                    self.index.token_count += 1
                    self.index.inverted_index.append(set())

                if self.index.token_map[token] not in token_index_set:
                    token_index_set[self.index.token_map[token]] = 0
                token_index_set[self.index.token_map[token]] += 1

            doc_i = self.index.node_id_map[ids_list[i]]
            self.index.doc_lens[doc_i] = len(tokens)

            # 重复
            if doc_i < len(self.index.doc_token_index):
                token_index_diff = set(self.index.doc_token_index[doc_i].keys()) - set(
                    token_index_set.keys()
                )
                for token_i in token_index_diff:
                    self.index.inverted_index[token_i].discard(doc_i)

                self.index.doc_token_index[doc_i] = token_index_set
            else:
                self.index.doc_token_index.append(token_index_set)

            for token_i in token_index_set:
                self.index.inverted_index[token_i].add(doc_i)


if __name__ == "__main__":
    start = datetime.datetime.now()
    print(f"{datetime.datetime.now()} Starting..")
    file = "/Users/feiyue/Documents/df_multi_col_data_media.csv"

    def load_file(file_name, store):
        node_list = []
        df = pd.read_csv(file_name, encoding="gb18030")
        for i, record in enumerate(df.to_dict(orient="records")):
            extra_info = {"row": i, "file_path": file}
            node_list.append(
                TextNode(id_=f"{i}", text=f"{record}", metadata=extra_info)
            )
        print(f"{datetime.datetime.now()} Load complete, start building index...")
        store.add_docs(node_list)

    store = PaiBm25Index("tmp/")

    load_file(file, store)

    print((datetime.datetime.now() - start))
    start2 = datetime.datetime.now()
    nodes = store.query("下雪的电影")
    print((datetime.datetime.now() - start2))
    for n in nodes:
        print(n.score, n.get_content())
