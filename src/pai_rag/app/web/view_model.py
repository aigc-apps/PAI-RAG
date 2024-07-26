from pydantic import BaseModel
from typing import Any, Dict
from collections import defaultdict
from pai_rag.app.web.ui_constants import (
    EMBEDDING_DIM_DICT,
    DEFAULT_EMBED_SIZE,
    DEFAULT_HF_EMBED_MODEL,
    LLM_MODEL_KEY_DICT,
    PROMPT_MAP,
)
import pandas as pd
import os
from datetime import datetime
import tempfile
import json


def recursive_dict():
    return defaultdict(recursive_dict)


def _transform_to_dict(config):
    for k, v in config.items():
        if isinstance(v, defaultdict):
            config[k] = _transform_to_dict(v)
    return dict(config)


class ViewModel(BaseModel):
    # embedding
    embed_source: str = "HuggingFace"
    embed_model: str = DEFAULT_HF_EMBED_MODEL
    embed_dim: int = 1024
    embed_api_key: str = None
    embed_batch_size: int = 10

    # llm
    llm: str = "PaiEas"
    llm_eas_url: str = None
    llm_eas_token: str = None
    llm_eas_model_name: str = "PAI-EAS-LLM"
    llm_api_key: str = None
    llm_api_model_name: str = None
    llm_temperature: float = 0.1

    # chunking
    parser_type: str = "Sentence"
    chunk_size: int = 500
    chunk_overlap: int = 20

    # reader
    reader_type: str = "SimpleDirectoryReader"
    enable_qa_extraction: bool = False
    enable_raptor: bool = False

    config_file: str = None

    vectordb_type: str = "FAISS"

    # AnalyticDB
    adb_ak: str = None
    adb_sk: str = None
    adb_region_id: str = None
    adb_instance_id: str = None
    adb_account: str = None
    adb_account_password: str = None
    adb_namespace: str = "pairag"
    adb_collection: str = "pairag_collection"
    adb_metrics: str = "cosine"

    # Hologres
    hologres_database: str = "pairag"
    hologres_table: str = "pairag"
    hologres_user: str = None
    hologres_password: str = None
    hologres_host: str = None
    hologres_port: int = 80
    hologres_pre_delete: bool = False

    # Faiss
    faiss_path: str = None

    # ElasticSearch
    es_url: str = None
    es_index: str = None
    es_user: str = None
    es_password: str = None

    # Milvus
    milvus_host: str = None
    milvus_port: int = None
    milvus_user: str = None
    milvus_password: str = None
    milvus_database: str = "pairag"
    milvus_collection_name: str = "pairagcollection"

    # retriever
    similarity_top_k: int = 5
    retrieval_mode: str = "hybrid"  # hybrid / embedding / keyword
    query_rewrite_n: int = 1

    # postprocessor
    reranker_type: str = (
        "simple-weighted-reranker"  # simple-weighted-reranker / model-based-reranker
    )
    reranker_model: str = "bge-reranker-base"  # bge-reranker-base / bge-reranker-large
    keyword_weight: float = 0.3
    vector_weight: float = 0.7
    similarity_threshold: float = None

    query_engine_type: str = "RetrieverQueryEngine"

    synthesizer_type: str = None

    text_qa_template: str = None

    def update(self, update_paras: Dict[str, Any]):
        attr_set = set(dir(self))
        for key, value in update_paras.items():
            if key in attr_set:
                setattr(self, key, value)

    @staticmethod
    def from_app_config(config):
        view_model = ViewModel()
        view_model.embed_source = config["embedding"].get(
            "source", view_model.embed_source
        )
        view_model.embed_model = config["embedding"].get(
            "model_name", view_model.embed_model
        )
        view_model.embed_api_key = config["embedding"].get(
            "api_key", view_model.embed_api_key
        )
        view_model.embed_batch_size = config["embedding"].get(
            "embed_batch_size", view_model.embed_batch_size
        )

        view_model.llm = config["llm"].get("source", view_model.llm)
        view_model.llm_eas_url = config["llm"].get("endpoint", view_model.llm_eas_url)
        view_model.llm_eas_token = config["llm"].get("token", view_model.llm_eas_token)
        view_model.llm_api_key = config["llm"].get("api_key", view_model.llm_api_key)
        view_model.llm_temperature = config["llm"].get(
            "temperature", view_model.llm_temperature
        )
        if view_model.llm.lower() == "paieas":
            view_model.llm_eas_model_name = config["llm"].get(
                "name", view_model.llm_eas_model_name
            )
        else:
            view_model.llm_api_model_name = config["llm"].get(
                "name", view_model.llm_api_model_name
            )

        view_model.vectordb_type = config["index"]["vector_store"].get(
            "type", view_model.vectordb_type
        )
        view_model.faiss_path = config["index"].get(
            "persist_path", view_model.faiss_path
        )
        if view_model.vectordb_type == "AnalyticDB":
            view_model.adb_ak = config["index"]["vector_store"]["ak"]
            view_model.adb_sk = config["index"]["vector_store"]["sk"]
            view_model.adb_region_id = config["index"]["vector_store"]["region_id"]
            view_model.adb_instance_id = config["index"]["vector_store"]["instance_id"]
            view_model.adb_account = config["index"]["vector_store"]["account"]
            view_model.adb_account_password = config["index"]["vector_store"][
                "account_password"
            ]
            view_model.adb_namespace = config["index"]["vector_store"]["namespace"]
            view_model.adb_collection = config["index"]["vector_store"]["collection"]
            view_model.adb_metrics = config["index"]["vector_store"].get(
                "metrics", "cosine"
            )

        elif view_model.vectordb_type == "Hologres":
            view_model.hologres_host = config["index"]["vector_store"]["host"]
            view_model.hologres_port = config["index"]["vector_store"]["port"]
            view_model.hologres_user = config["index"]["vector_store"]["user"]
            view_model.hologres_password = config["index"]["vector_store"]["password"]
            view_model.hologres_database = config["index"]["vector_store"]["database"]
            view_model.hologres_table = config["index"]["vector_store"]["table_name"]
            view_model.hologres_pre_delete = config["index"]["vector_store"].get(
                "pre_delete_table", False
            )

        elif view_model.vectordb_type == "ElasticSearch":
            view_model.es_index = config["index"]["vector_store"]["es_index"]
            view_model.es_url = config["index"]["vector_store"]["es_url"]
            view_model.es_user = config["index"]["vector_store"]["es_user"]
            view_model.es_password = config["index"]["vector_store"]["es_password"]

        elif view_model.vectordb_type == "Milvus":
            view_model.milvus_host = config["index"]["vector_store"]["host"]
            view_model.milvus_port = config["index"]["vector_store"]["port"]
            view_model.milvus_user = config["index"]["vector_store"]["user"]
            view_model.milvus_password = config["index"]["vector_store"]["password"]
            view_model.milvus_database = config["index"]["vector_store"]["database"]
            view_model.milvus_collection_name = config["index"]["vector_store"][
                "collection_name"
            ]

        view_model.parser_type = config["node_parser"]["type"]
        view_model.chunk_size = config["node_parser"]["chunk_size"]
        view_model.chunk_overlap = config["node_parser"]["chunk_overlap"]

        view_model.reader_type = config["data_reader"].get(
            "type", view_model.reader_type
        )
        view_model.enable_qa_extraction = config["data_reader"].get(
            "enable_qa_extraction", view_model.enable_qa_extraction
        )
        view_model.enable_raptor = config["data_reader"].get(
            "enable_raptor", view_model.enable_raptor
        )

        view_model.similarity_top_k = config["retriever"].get("similarity_top_k", 5)
        if config["retriever"]["retrieval_mode"] == "hybrid":
            view_model.retrieval_mode = "Hybrid"
            view_model.query_rewrite_n = config["retriever"]["query_rewrite_n"]
        elif config["retriever"]["retrieval_mode"] == "embedding":
            view_model.retrieval_mode = "Embedding Only"
        elif config["retriever"]["retrieval_mode"] == "keyword":
            view_model.retrieval_mode = "Keyword Only"

        reranker_type = config["postprocessor"].get(
            "reranker_type", "simple-weighted-reranker"
        )
        similarity_threshold = config["postprocessor"].get("similarity_threshold", None)
        view_model.similarity_threshold = (
            similarity_threshold if similarity_threshold > 0 else None
        )

        if reranker_type == "simple-weighted-reranker":
            view_model.reranker_type = "simple-weighted-reranker"
            vector_weight = config["postprocessor"].get("vector_weight", 0.7)
            view_model.vector_weight = float(vector_weight)
            keyword_weight = config["postprocessor"].get("keyword_weight", 0.3)
            view_model.keyword_weight = float(keyword_weight)

        elif reranker_type == "model-based-reranker":
            view_model.reranker_type = "model-based-reranker"
            view_model.reranker_model = config["postprocessor"].get(
                "reranker_model", "bge-reranker-base"
            )

        view_model.synthesizer_type = config["synthesizer"].get(
            "type", "SimpleSummarize"
        )
        view_model.text_qa_template = config["synthesizer"].get(
            "text_qa_template", None
        )

        if config["query_engine"]["type"] == "TransformQueryEngine":
            view_model.query_engine_type = "TransformQueryEngine"
        else:
            view_model.query_engine_type = "RetrieverQueryEngine"

        return view_model

    def to_app_config(self):
        config = recursive_dict()

        config["embedding"]["source"] = self.embed_source
        config["embedding"]["model_name"] = self.embed_model
        config["embedding"]["api_key"] = self.embed_api_key
        config["embedding"]["embed_batch_size"] = int(self.embed_batch_size)

        config["llm"]["source"] = self.llm
        config["llm"]["endpoint"] = self.llm_eas_url
        config["llm"]["token"] = self.llm_eas_token
        config["llm"]["api_key"] = self.llm_api_key
        config["llm"]["temperature"] = self.llm_temperature
        if self.llm.lower() == "paieas":
            config["llm"]["name"] = self.llm_eas_model_name
        else:
            config["llm"]["name"] = self.llm_api_model_name

        config["index"]["vector_store"]["type"] = self.vectordb_type
        config["index"]["persist_path"] = self.faiss_path

        config["node_parser"]["type"] = self.parser_type
        config["node_parser"]["chunk_size"] = int(self.chunk_size)
        config["node_parser"]["chunk_overlap"] = int(self.chunk_overlap)

        config["data_reader"]["enable_qa_extraction"] = self.enable_qa_extraction
        config["data_reader"]["enable_raptor"] = self.enable_raptor
        config["data_reader"]["type"] = self.reader_type

        if self.vectordb_type == "Hologres":
            config["index"]["vector_store"]["host"] = self.hologres_host
            config["index"]["vector_store"]["port"] = self.hologres_port
            config["index"]["vector_store"]["user"] = self.hologres_user
            config["index"]["vector_store"]["password"] = self.hologres_password
            config["index"]["vector_store"]["database"] = self.hologres_database
            config["index"]["vector_store"]["table_name"] = self.hologres_table
            config["index"]["vector_store"][
                "pre_delete_table"
            ] = self.hologres_pre_delete

        elif self.vectordb_type == "AnalyticDB":
            config["index"]["vector_store"]["ak"] = self.adb_ak
            config["index"]["vector_store"]["sk"] = self.adb_sk
            config["index"]["vector_store"]["region_id"] = self.adb_region_id
            config["index"]["vector_store"]["instance_id"] = self.adb_instance_id
            config["index"]["vector_store"]["account"] = self.adb_account
            config["index"]["vector_store"][
                "account_password"
            ] = self.adb_account_password
            config["index"]["vector_store"]["namespace"] = self.adb_namespace
            config["index"]["vector_store"]["collection"] = self.adb_collection
            config["index"]["vector_store"]["metrics"] = self.adb_metrics

        elif self.vectordb_type == "ElasticSearch":
            config["index"]["vector_store"]["es_index"] = self.es_index
            config["index"]["vector_store"]["es_url"] = self.es_url
            config["index"]["vector_store"]["es_user"] = self.es_user
            config["index"]["vector_store"]["es_password"] = self.es_password

        elif self.vectordb_type == "Milvus":
            config["index"]["vector_store"]["host"] = self.milvus_host
            config["index"]["vector_store"]["port"] = self.milvus_port
            config["index"]["vector_store"]["user"] = self.milvus_user
            config["index"]["vector_store"]["data"] = self.milvus_password
            config["index"]["vector_store"]["database"] = self.milvus_password
            config["index"]["vector_store"]["password"] = self.milvus_password
            config["index"]["vector_store"]["database"] = self.milvus_database
            config["index"]["vector_store"][
                "collection_name"
            ] = self.milvus_collection_name

        config["retriever"]["similarity_top_k"] = self.similarity_top_k
        if self.retrieval_mode == "Hybrid":
            config["retriever"]["retrieval_mode"] = "hybrid"
            config["retriever"]["query_rewrite_n"] = self.query_rewrite_n
        elif self.retrieval_mode == "Embedding Only":
            config["retriever"]["retrieval_mode"] = "embedding"
        elif self.retrieval_mode == "Keyword Only":
            config["retriever"]["retrieval_mode"] = "keyword"

        config["postprocessor"]["reranker_type"] = self.reranker_type
        config["postprocessor"]["reranker_model"] = self.reranker_model
        config["postprocessor"]["keyword_weight"] = self.keyword_weight
        config["postprocessor"]["vector_weight"] = self.vector_weight
        config["postprocessor"]["similarity_threshold"] = self.similarity_threshold
        config["postprocessor"]["top_n"] = self.similarity_top_k

        config["synthesizer"]["type"] = self.synthesizer_type
        config["synthesizer"]["text_qa_template"] = self.text_qa_template
        if self.query_engine_type == "TransformQueryEngine":
            config["query_engine"]["type"] = "TransformQueryEngine"
        else:
            config["query_engine"]["type"] = "RetrieverQueryEngine"

        return _transform_to_dict(config)

    def get_local_generated_qa_file(self):
        DEFALUT_EVAL_PATH = "localdata/evaluation"
        qa_dataset_path = os.path.join(DEFALUT_EVAL_PATH, "qa_dataset.json")
        if os.path.exists(qa_dataset_path):
            tmpdir = tempfile.mkdtemp()
            with open(qa_dataset_path, "r", encoding="utf-8") as file:
                qa_content = json.load(file)
            outputPath = os.path.join(tmpdir, "qa_dataset.json")
            with open(outputPath, "w", encoding="utf-8") as f:
                json.dump(qa_content, f, ensure_ascii=False, indent=4)
            return outputPath, qa_content["examples"][0:5]
        else:
            return None, None

    def get_local_evaluation_result_file(self, type):
        DEFALUT_EVAL_PATH = "localdata/evaluation"
        output_path = os.path.join(DEFALUT_EVAL_PATH, f"batch_eval_results_{type}.xlsx")
        if type == "retrieval":
            if os.path.exists(output_path):
                modification_time = os.path.getmtime(output_path)
                formatted_time = datetime.fromtimestamp(modification_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                df = pd.read_excel(output_path)
                retrieval_pd_results = {
                    "Metrics": ["HitRate", "MRR", "LastModified"],
                    "Value": [df["hit_rate"].mean(), df["mrr"].mean(), formatted_time],
                }
            else:
                retrieval_pd_results = {
                    "Metrics": ["HitRate", "MRR", "LastModified"],
                    "Value": [None, None, None],
                }
            return pd.DataFrame(retrieval_pd_results)
        elif type == "response":
            if os.path.exists(output_path):
                modification_time = os.path.getmtime(output_path)
                formatted_time = datetime.fromtimestamp(modification_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                df = pd.read_excel(output_path)
                response_pd_results = {
                    "Metrics": [
                        "Faithfulness",
                        "Correctness",
                        "SemanticSimilarity",
                        "LastModified",
                    ],
                    "Value": [
                        df["faithfulness_score"].mean(),
                        df["correctness_score"].mean(),
                        df["semantic_similarity_score"].mean(),
                        formatted_time,
                    ],
                }
            else:
                response_pd_results = {
                    "Metrics": [
                        "Faithfulness",
                        "Correctness",
                        "SemanticSimilarity",
                        "LastModified",
                    ],
                    "Value": [None, None, None, None],
                }
            return pd.DataFrame(response_pd_results)
        else:
            raise ValueError(f"Not supported the evaluation type {type}")

    def to_component_settings(self) -> Dict[str, Dict[str, Any]]:
        settings = {}
        settings["embed_source"] = {"value": self.embed_source}
        settings["embed_model"] = {
            "value": self.embed_model,
            "visible": self.embed_source == "HuggingFace",
        }
        settings["embed_dim"] = {
            "value": EMBEDDING_DIM_DICT.get(self.embed_model, DEFAULT_EMBED_SIZE)
            if self.embed_source == "HuggingFace"
            else DEFAULT_EMBED_SIZE
        }
        settings["embed_batch_size"] = {"value": self.embed_batch_size}

        settings["llm"] = {"value": self.llm}
        settings["llm_eas_url"] = {"value": self.llm_eas_url}
        settings["llm_eas_token"] = {"value": self.llm_eas_token}
        settings["llm_eas_model_name"] = {"value": self.llm_eas_model_name}
        settings["llm_api_model_name"] = {
            "value": self.llm_api_model_name,
            "choices": LLM_MODEL_KEY_DICT.get(self.llm, []),
            "visible": self.llm.lower() != "paieas",
        }
        settings["chunk_size"] = {"value": self.chunk_size}
        settings["chunk_overlap"] = {"value": self.chunk_overlap}
        settings["enable_qa_extraction"] = {"value": self.enable_qa_extraction}
        settings["enable_raptor"] = {"value": self.enable_raptor}

        # retrieval and rerank
        settings["retrieval_mode"] = {"value": self.retrieval_mode}
        settings["reranker_type"] = {"value": self.reranker_type}
        settings["similarity_top_k"] = {"value": self.similarity_top_k}
        settings["reranker_model"] = {"value": self.reranker_model}
        settings["vector_weight"] = {"value": self.vector_weight}
        settings["keyword_weight"] = {"value": self.keyword_weight}
        settings["similarity_threshold"] = {"value": self.similarity_threshold}

        prm_type = PROMPT_MAP.get(self.text_qa_template, "Custom")
        settings["prm_type"] = {"value": prm_type}
        settings["text_qa_template"] = {"value": self.text_qa_template}

        settings["vectordb_type"] = {"value": self.vectordb_type}

        # adb
        settings["adb_ak"] = {"value": self.adb_ak}
        settings["adb_sk"] = {"value": self.adb_sk}
        settings["adb_region_id"] = {"value": self.adb_region_id}
        settings["adb_account"] = {"value": self.adb_account}
        settings["adb_account_password"] = {"value": self.adb_account_password}
        settings["adb_namespace"] = {"value": self.adb_namespace}
        settings["adb_instance_id"] = {"value": self.adb_instance_id}
        settings["adb_collection"] = {"value": self.adb_collection}

        # hologres
        settings["hologres_host"] = {"value": self.hologres_host}
        settings["hologres_database"] = {"value": self.hologres_database}
        settings["hologres_port"] = {"value": self.hologres_port}
        settings["hologres_user"] = {"value": self.hologres_user}
        settings["hologres_password"] = {"value": self.hologres_password}
        settings["hologres_table"] = {"value": self.hologres_table}
        settings["hologres_pre_delete"] = {"value": self.hologres_pre_delete}

        # elasticsearch
        settings["es_url"] = {"value": self.es_url}
        settings["es_index"] = {"value": self.es_index}
        settings["es_user"] = {"value": self.es_user}
        settings["es_password"] = {"value": self.es_password}

        # milvus
        settings["milvus_host"] = {"value": self.milvus_host}
        settings["milvus_port"] = {"value": self.milvus_port}
        settings["milvus_database"] = {"value": self.milvus_database}
        settings["milvus_user"] = {"value": self.milvus_user}
        settings["milvus_password"] = {"value": self.milvus_password}
        settings["milvus_collection_name"] = {"value": self.milvus_collection_name}

        # faiss
        settings["faiss_path"] = {"value": self.faiss_path}

        # evaluation
        if self.vectordb_type == "FAISS":
            qa_dataset_path, qa_dataset_res = self.get_local_generated_qa_file()
            settings["qa_dataset_file"] = {"value": qa_dataset_path}
            settings["qa_dataset_json_text"] = {"value": qa_dataset_res}
            settings["eval_retrieval_res"] = {
                "value": self.get_local_evaluation_result_file(type="retrieval")
            }
            settings["eval_response_res"] = {
                "value": self.get_local_evaluation_result_file(type="response")
            }

        return settings
