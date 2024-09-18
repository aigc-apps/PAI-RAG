import datetime
import json
import os
from typing import Any, Dict, List
from fastapi.concurrency import run_in_threadpool
from llama_index.core import Settings
from llama_index.core.schema import TextNode, ImageNode, ImageDocument
from llama_index.llms.huggingface import HuggingFaceLLM

from pai_rag.integrations.nodeparsers.base import MarkdownNodeParser
from pai_rag.integrations.extractors.html_qa_extractor import HtmlQAExtractor
from pai_rag.integrations.extractors.text_qa_extractor import TextQAExtractor
from pai_rag.modules.nodeparser.node_parser import node_id_hash
from pai_rag.data.open_dataset import MiraclOpenDataSet, DuRetrievalDataSet
from llama_index.core.schema import BaseNode

import logging
import re

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_QA_MODEL_PATH = "./model_repository/qwen_1.8b"
DOC_TYPES_DO_NOT_NEED_CHUNKING = set(
    [".csv", ".xlsx", ".xls", ".htm", ".html", ".jsonl"]
)
IMAGE_FILE_TYPES = set([".jpg", ".jpeg", ".png"])

IMAGE_URL_REGEX = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:jpg|jpeg|png)",
    re.IGNORECASE,
)


class OssDataLoader:
    """
    OssDataLoader:
    Load data with corresponding data readers according to config.
    """

    def __init__(
        self,
        datareader_factory,
        node_parser,
        index,
        bm25_index,
        oss_cache,
        node_enhance,
        use_local_qa_model=False,
    ):
        self.datareader_factory = datareader_factory
        self.node_parser = node_parser
        self.oss_cache = oss_cache
        self.index = index
        self.bm25_index = bm25_index
        self.node_enhance = node_enhance

        if use_local_qa_model:
            # API暂不支持此选项
            self.qa_llm = HuggingFaceLLM(
                model_name=DEFAULT_LOCAL_QA_MODEL_PATH,
                tokenizer_name=DEFAULT_LOCAL_QA_MODEL_PATH,
            )
        else:
            self.qa_llm = Settings.llm

        html_extractor = HtmlQAExtractor(llm=self.qa_llm)
        txt_extractor = TextQAExtractor(llm=self.qa_llm)

        self.extractors = [html_extractor, txt_extractor]

        logger.info("OssDataLoader initialized.")

    def _extract_file_type(self, metadata: Dict[str, Any]):
        file_name = metadata.get("file_name", "dummy.txt")
        return os.path.splitext(file_name)[1]

    def _get_oss_files(self):
        files = []
        if self.oss_cache:
            object_list = self.oss_cache.list_objects()
            oss_file_path_dir = "localdata/oss_tmp"
            if not os.path.exists(oss_file_path_dir):
                os.makedirs(oss_file_path_dir)
            for oss_obj in object_list:
                if oss_obj.key[:-1] != self.oss_cache.prefix:
                    try:
                        set_public = self.oss_cache.put_object_acl(
                            oss_obj.key, "public-read"
                        )
                    except Exception:
                        logger.error(f"Failed to set_public document {oss_obj.key}")
                    if set_public:
                        save_filename = os.path.join(oss_file_path_dir, oss_obj.key)
                        self.oss_cache.get_object_to_file(
                            key=oss_obj.key, filename=save_filename
                        )
                        files.append(save_filename)
                    else:
                        logger.error(f"Failed to load document {oss_obj.key}")
        return files

    def _filter_text_nodes(self, nodes: List[BaseNode]):
        filtered_nodes = []
        text_seen = set()
        text_seen.update(
            node.text.strip()
            for node in nodes
            if isinstance(node, TextNode)
            and node.metadata.get("image_url_list_str") is not None
        )
        for node in nodes:
            if (
                isinstance(node, TextNode)
                and node.metadata.get("image_url_list_str") is None
                and node.text.strip() in text_seen
            ):
                continue
            filtered_nodes.append(node)
        return filtered_nodes

    def _get_nodes(
        self,
        file_path: str | List[str],
        filter_pattern: str,
        enable_qa_extraction: bool,
    ):
        tmp_index_doc = self.index.vector_index._docstore.docs
        seen_files = set(
            [_doc.metadata.get("file_name") for _, _doc in tmp_index_doc.items()]
        )
        filter_pattern = filter_pattern or "*"
        if isinstance(file_path, list):
            input_files = [f for f in file_path if os.path.isfile(f)]
        elif isinstance(file_path, str) and os.path.isdir(file_path):
            import pathlib

            directory = pathlib.Path(file_path)
            input_files = [
                f for f in directory.rglob(filter_pattern) if os.path.isfile(f)
            ]
        else:
            input_files = [file_path]

        if len(input_files) == 0:
            return

        # 检查文件名是否已经在seen_files中，如果在，则跳过当前文件
        new_input_files = []
        for input_file in input_files:
            if os.path.basename(input_file) in seen_files:
                print(
                    f"[RagOssDataLoader] {os.path.basename(input_file)} already exists, skip it."
                )
                continue
            new_input_files.append(input_file)
        if len(new_input_files) == 0:
            return
        print(f"[RagOssDataLoader] {len(new_input_files)} files will be loaded.")

        data_reader = self.datareader_factory.get_reader(new_input_files)
        docs = data_reader.load_data()
        logger.info(f"[DataReader] Loaded {len(docs)} docs.")
        nodes = []

        doc_cnt_map = {}
        for doc in docs:
            doc_type = self._extract_file_type(doc.metadata)
            doc.metadata["file_path"] = os.path.basename(doc.metadata["file_path"])
            doc.metadata["file_url"] = self.oss_cache.get_obj_key_url(
                doc.metadata["file_path"]
            )
            doc_key = f"""{doc.metadata.get("file_path", "dummy")}"""
            if doc_key not in doc_cnt_map:
                doc_cnt_map[doc_key] = 0

            if isinstance(doc, ImageDocument):
                node_id = node_id_hash(doc_cnt_map[doc_key], doc)
                doc_cnt_map[doc_key] += 1
                nodes.append(
                    ImageNode(
                        id_=node_id, image_url=doc.image_url, metadata=doc.metadata
                    )
                )
            elif doc_type in DOC_TYPES_DO_NOT_NEED_CHUNKING:
                doc_key = f"""{doc.metadata.get("file_path", "dummy")}"""
                doc_cnt_map[doc_key] += 1
                node_id = node_id_hash(doc_cnt_map[doc_key], doc)
                nodes.append(
                    TextNode(id_=node_id, text=doc.text, metadata=doc.metadata)
                )
            elif doc_type == ".md" or doc_type == ".pdf":
                md_node_parser = MarkdownNodeParser(id_func=node_id_hash)
                tmp_nodes = md_node_parser.get_nodes_from_documents([doc])
                for node in tmp_nodes:
                    node.id_ = node_id_hash(doc_cnt_map[doc_key], doc)
                    doc_cnt_map[doc_key] += 1
                    nodes.append(node)
            else:
                nodes.extend(self.node_parser.get_nodes_from_documents([doc]))

        for node in nodes:
            node.excluded_embed_metadata_keys.append("file_path")
            node.excluded_embed_metadata_keys.append(
                "image_url" if isinstance(node, ImageNode) else "image_url_list_str"
            )
            node.excluded_embed_metadata_keys.append("total_pages")
            node.excluded_embed_metadata_keys.append("source")

        logger.info(f"[DataReader] Split into {len(nodes)} nodes.")

        nodes = self._filter_text_nodes(nodes)
        # QA metadata extraction
        if enable_qa_extraction:
            qa_nodes = []

            for extractor in self.extractors:
                metadata_list = extractor.extract(nodes)
                for i, node in enumerate(nodes):
                    qa_extraction_result = metadata_list[i].get(
                        "qa_extraction_result", {}
                    )
                    q_cnt = 0
                    metadata = node.metadata
                    for q, a in qa_extraction_result.items():
                        metadata["answer"] = a
                        qa_nodes.append(
                            TextNode(
                                id_=f"{node.id_}_{q_cnt}", text=q, metadata=metadata
                            )
                        )
                        q_cnt += 1
            for node in qa_nodes:
                node.excluded_embed_metadata_keys.append("answer")
                node.excluded_llm_metadata_keys.append("question")
            nodes.extend(qa_nodes)

        return nodes

    def load(
        self,
        filter_pattern: str,
        enable_qa_extraction: bool,
        enable_raptor: bool,
    ):
        file_path = self._get_oss_files()
        nodes = self._get_nodes(file_path, filter_pattern, enable_qa_extraction)

        if not nodes:
            logger.warning("[DataReader] no nodes parsed.")
            return

        logger.info("[DataReader] Start inserting to index.")

        if enable_raptor:
            nodes_with_embeddings = self.node_enhance(nodes=nodes)
            self.index.vector_index.insert_nodes(nodes_with_embeddings)

            logger.info(
                f"Inserted {len(nodes)} and enhanced {len(nodes_with_embeddings)-len(nodes)} nodes successfully."
            )
        else:
            self.index.vector_index.insert_nodes(nodes)
            logger.info(f"Inserted {len(nodes)} nodes successfully.")

        self.index.vector_index.storage_context.persist(
            persist_dir=self.index.persist_path
        )

        index_metadata_file = os.path.join(self.index.persist_path, "index.metadata")
        if self.bm25_index:
            self.bm25_index.add_docs(nodes)
            metadata_str = json.dumps({"lastUpdated": f"{datetime.datetime.now()}"})
            with open(index_metadata_file, "w") as wf:
                wf.write(metadata_str)

        return

    async def aload(
        self,
        filter_pattern: str,
        enable_qa_extraction: bool,
        enable_raptor: bool,
    ):
        file_path = self._get_oss_files()
        nodes = await run_in_threadpool(
            lambda: self._get_nodes(file_path, filter_pattern, enable_qa_extraction)
        )
        if not nodes:
            logger.info("[DataReader] could not find files")
            return

        logger.info("[DataReader] Start inserting to index.")

        if enable_raptor:
            nodes_with_embeddings = await self.node_enhance.acall(nodes=nodes)
            self.index.vector_index.insert_nodes(nodes_with_embeddings)

            logger.info(
                f"Async inserted {len(nodes)} and enhanced {len(nodes_with_embeddings)-len(nodes)} nodes successfully."
            )

        else:
            self.index.vector_index.insert_nodes(nodes)
            logger.info(f"Inserted {len(nodes)} nodes successfully.")

        self.index.vector_index.storage_context.persist(
            persist_dir=self.index.persist_path
        )

        index_metadata_file = os.path.join(self.index.persist_path, "index.metadata")
        if self.bm25_index:
            await run_in_threadpool(lambda: self.bm25_index.add_docs(nodes))
            metadata_str = json.dumps({"lastUpdated": f"{datetime.datetime.now()}"})
            with open(index_metadata_file, "w") as wf:
                wf.write(metadata_str)

        return

    def load_eval_data(self, name: str):
        logger.info("[DataReader-Evaluation Dataset]")
        if name == "miracl":
            miracl_dataset = MiraclOpenDataSet()
            miracl_nodes, _ = miracl_dataset.load_related_corpus()
            nodes = []
            for node in miracl_nodes:
                node_metadata = {
                    "title": node[2],
                    "file_path": node[3],
                    "file_name": node[3],
                }
                nodes.append(
                    TextNode(id_=node[0], text=node[1], metadata=node_metadata)
                )

            print(f"[DataReader-Evaluation Dataset] Split into {len(nodes)} nodes.")

            print("[DataReader-Evaluation Dataset] Start inserting to index.")

            self.index.vector_index.insert_nodes(nodes)
            self.index.vector_index.storage_context.persist(
                persist_dir=self.index.persist_path
            )

            index_metadata_file = os.path.join(
                self.index.persist_path, "index.metadata"
            )
            if self.bm25_index:
                self.bm25_index.add_docs(nodes)
                metadata_str = json.dumps({"lastUpdated": f"{datetime.datetime.now()}"})
                with open(index_metadata_file, "w") as wf:
                    wf.write(metadata_str)

            print(
                f"[DataReader-Evaluation Dataset] Inserted {len(nodes)} nodes successfully."
            )
            return
        elif name == "duretrieval":
            duretrieval_dataset = DuRetrievalDataSet()
            miracl_nodes, _, _ = duretrieval_dataset.load_related_corpus()
            nodes = []
            for node in miracl_nodes:
                node_metadata = {
                    "file_path": node[2],
                    "file_name": node[2],
                }
                nodes.append(
                    TextNode(id_=node[0], text=node[1], metadata=node_metadata)
                )

            print(f"[DataReader-Evaluation Dataset] Split into {len(nodes)} nodes.")

            print("[DataReader-Evaluation Dataset] Start inserting to index.")

            self.index.vector_index.insert_nodes(nodes)
            self.index.vector_index.storage_context.persist(
                persist_dir=self.index.persist_path
            )

            index_metadata_file = os.path.join(
                self.index.persist_path, "index.metadata"
            )
            if self.bm25_index:
                self.bm25_index.add_docs(nodes)
                metadata_str = json.dumps({"lastUpdated": f"{datetime.datetime.now()}"})
                with open(index_metadata_file, "w") as wf:
                    wf.write(metadata_str)

            print(
                f"[DataReader-Evaluation Dataset] Inserted {len(nodes)} nodes successfully."
            )
            return
        else:
            raise ValueError(f"Not supported eval dataset name with {name}")
