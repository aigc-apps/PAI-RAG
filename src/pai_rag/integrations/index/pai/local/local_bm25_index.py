import logging
import os
import pickle
import json
import numpy as np
from typing import Callable, List, cast, Dict
from llama_index.core.schema import BaseNode, TextNode, NodeWithScore
from pai_rag.utils.tokenizer import jieba_tokenizer
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


# Handle module mismatch issue.
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "pai_rag.modules.index.pai_bm25_index":
            renamed_module = "pai_rag.integrations.index.pai.local.local_bm25_index"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


class LocalBm25Index:
    def __init__(self):
        self.doc_count: int = 0
        self.token_count: int = 0
        self.doc_lens: np.ndarray = np.array([])
        self.doc_token_index: List[Dict[str, int]] = []
        self.inverted_index: List[set[int]] = []
        self.token_map: Dict[str, int] = {}
        self.node_id_map: Dict[str, int] = {}


class LocalBm25IndexStore:
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

        self.persist_path = persist_path
        self.data_path = os.path.join(persist_path, DEFAULT_STORE_DIR)
        self.parts_path = os.path.join(self.data_path, DEFAULT_FILE_PART_DIR)
        self.index_file = os.path.join(self.data_path, DEFAULT_INDEX_FILE)
        self.index_matrix_file = os.path.join(self.data_path, DEFAULT_INDEX_MATRIX_FILE)

        self.workers = workers
        self.tokenizer = tokenizer or jieba_tokenizer

        logger.info(f"Start loading local BM25 index @ {self.data_path}!")
        self.reload()
        logger.info(f"Finished loading BM25 index @ {self.data_path}!")

    def reload(self):
        if os.path.exists(self.parts_path):
            with open(self.index_file, "rb") as f:
                self.index: LocalBm25Index = renamed_load(f)
            with open(self.index_matrix_file, "rb") as f:
                self.index_matrix = pickle.load(f)
        else:
            self.index = LocalBm25Index()
            self.index_matrix = None
            self.token_map = {}
        self.doc_cache = {}

    def split_doc(self, text_list: List[str], tokenizer: Callable):
        tokens_list = []
        for text in text_list:
            tokens = tokenizer(text)
            tokens_list.append(tokens)

        logger.info(f"Finished {len(text_list)} docs.")
        return tokens_list

    def persist(self, doc_index_list, doc_list):
        os.makedirs(self.parts_path, exist_ok=True)

        with open(self.index_file, "wb") as wf:
            pickle.dump(self.index, wf)

        with open(self.index_matrix_file, "wb") as wf:
            pickle.dump(self.index_matrix, wf)

        doc_idx_map = dict(zip(doc_index_list, doc_list))
        bucket_size = DEFAULT_PART_SIZE
        pre_bucket = -1
        current_batch = []
        part_file_name = "default.part"
        for i in sorted(doc_idx_map):
            bucket = int(i / bucket_size)
            if bucket != pre_bucket:
                if current_batch:
                    part_file_name = os.path.join(
                        self.parts_path, f"{pre_bucket+1:06}.part"
                    )
                    with open(part_file_name, "w") as wf:
                        wf.write("\n".join(current_batch))
                        current_batch = []

                part_file_name = os.path.join(self.parts_path, f"{bucket+1:06}.part")
                if os.path.exists(part_file_name):
                    current_batch = open(part_file_name, "r").readlines()
                    current_batch = [line.strip() for line in current_batch]

                pre_bucket = bucket
            doc_i = i % bucket_size

            if doc_i < len(current_batch):
                current_batch[i % bucket_size] == doc_idx_map[i]
            else:
                current_batch.append(doc_idx_map[i])

        if current_batch:
            part_file_name = os.path.join(self.parts_path, f"{pre_bucket+1:06}.part")
            with open(part_file_name, "w") as wf:
                wf.write("\n".join(current_batch))
        logger.info("Write index succeed!")

    def add_docs(self, nodes: List[BaseNode]):
        node_index_list = []
        text_list = []
        id_list = []
        metadata_list = []

        for node in nodes:
            if isinstance(node, TextNode):
                text_node = cast(TextNode, node)
                if not text_node.get_content():
                    # skip empty nodes
                    continue
                node_id = text_node.node_id
                id_list.append(node_id)
                metadata_list.append(text_node.metadata)
                text_list.append(text_node.get_content())

                if node_id not in self.index.node_id_map:
                    node_index = self.index.doc_count
                    self.index.doc_count += 1
                    self.index.node_id_map[node_id] = node_index
                node_index_list.append(self.index.node_id_map[node_id])
                self.doc_cache[self.index.node_id_map[node_id]] = node
            else:
                # don't handle image or graph node
                pass

        if not id_list:
            return

        pad_size = self.index.doc_count - len(self.index.doc_lens)
        self.index.doc_lens = np.lib.pad(
            self.index.doc_lens, (0, pad_size), "constant", constant_values=(0)
        )

        tokens_list = self.split_doc(text_list, self.tokenizer)
        self.process_token_list(tokens_list, id_list)

        self.construct_index_matrix()

        doc_list = [
            json.dumps(
                {"id": id_list[i], "text": text_list[i], "metadata": metadata_list[i]},
                ensure_ascii=False,
            )
            for i in range(len(text_list))
        ]
        self.persist(node_index_list, doc_list)

        logger.info("Successfully write to BM25 index.")
        return

    def construct_index_matrix(self):
        # m * n matrix
        m = self.index.doc_count
        n = self.index.token_count
        if m == 0 or n == 0:
            return

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
        filtered_doc_indexes = [idx for idx in doc_indexes if idx not in self.doc_cache]
        if len(filtered_doc_indexes) == 0:
            return [self.doc_cache[idx] for idx in doc_indexes]

        node_indexes = filtered_doc_indexes.copy()
        node_indexes.sort()
        bucket_size = DEFAULT_PART_SIZE
        batch_ids = []
        pre_bucket = -1
        for i in node_indexes:
            bucket = int(i / bucket_size)
            if bucket != pre_bucket:
                if batch_ids:
                    batch_nodes = self.load_batch_from_part_file(batch_ids, bucket)
                    self.doc_cache.update(zip(batch_ids, batch_nodes))
                    batch_ids = []
                pre_bucket = bucket
            batch_ids.append(i)

        if batch_ids:
            batch_nodes = self.load_batch_from_part_file(batch_ids, bucket)
            self.doc_cache.update(zip(batch_ids, batch_nodes))

        return [self.doc_cache[i] for i in doc_indexes]

    def query(
        self, query_str: str, top_n: int = 5, normalize: bool = False
    ) -> List[NodeWithScore]:
        results: List[NodeWithScore] = []
        if self.index_matrix is None:
            return results

        tokens = self.tokenizer(query_str)
        query_vec = np.zeros(self.index.token_count)
        for token in tokens:
            if token in self.index.token_map:
                query_vec[self.index.token_map[token]] += 1
        doc_scores = self.index_matrix.multiply(query_vec).sum(axis=1).getA1()
        doc_indexes = doc_scores.argsort()[::-1][:top_n]
        text_nodes: List[TextNode] = self.load_docs_with_index(doc_indexes)
        for i, node in enumerate(text_nodes):
            results.append(
                NodeWithScore(
                    node=node,
                    score=doc_scores[doc_indexes[i]],
                )
            )

        if normalize and len(results) > 0:
            bm25_scores = [node.score for node in results]
            max_score = max(bm25_scores)
            if max_score > 0:
                for node_with_score in results:
                    node_with_score.score = node_with_score.score / max_score

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
