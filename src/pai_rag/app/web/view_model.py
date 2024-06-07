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
    llm_eas_model_name: str = None
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

    similarity_top_k: int = 5
    # similarity_cutoff: float = 0.3
    rerank_model: str = "no-reranker"
    retrieval_mode: str = "hybrid"  # hybrid / embedding / keyword
    query_engine_type: str = "RetrieverQueryEngine"
    BM25_weight: float = 0.5
    vector_weight: float = 0.5
    fusion_mode: str = "reciprocal_rerank"  # [simple, reciprocal_rerank, dist_based_score, relative_score]
    query_rewrite_n: int = 1

    synthesizer_type: str = None

    text_qa_template: str = None

    def update(self, update_paras: Dict[str, Any]):
        attr_set = set(dir(self))
        for key, value in update_paras.items():
            if key in attr_set:
                setattr(self, key, value)

    def sync_app_config(self, config):
        self.embed_source = config["embedding"].get("source", self.embed_source)
        self.embed_model = config["embedding"].get("model_name", self.embed_model)
        self.embed_api_key = config["embedding"].get("api_key", self.embed_api_key)
        self.embed_batch_size = config["embedding"].get(
            "embed_batch_size", self.embed_batch_size
        )

        self.llm = config["llm"].get("source", self.llm)
        self.llm_eas_url = config["llm"].get("endpoint", self.llm_eas_url)
        self.llm_eas_token = config["llm"].get("token", self.llm_eas_token)
        self.llm_api_key = config["llm"].get("api_key", self.llm_api_key)
        self.llm_temperature = config["llm"].get("temperature", self.llm_temperature)
        if self.llm == "PaiEAS":
            self.llm_eas_model_name = config["llm"].get("name", self.llm_eas_model_name)
        else:
            self.llm_api_model_name = config["llm"].get("name", self.llm_api_model_name)

        self.vectordb_type = config["index"]["vector_store"].get(
            "type", self.vectordb_type
        )
        self.faiss_path = config["index"].get("persist_path", self.faiss_path)
        if self.vectordb_type == "AnalyticDB":
            self.adb_ak = config["index"]["vector_store"]["ak"]
            self.adb_sk = config["index"]["vector_store"]["sk"]
            self.adb_region_id = config["index"]["vector_store"]["region_id"]
            self.adb_instance_id = config["index"]["vector_store"]["instance_id"]
            self.adb_account = config["index"]["vector_store"]["account"]
            self.adb_account_password = config["index"]["vector_store"][
                "account_password"
            ]
            self.adb_namespace = config["index"]["vector_store"]["namespace"]
            self.adb_collection = config["index"]["vector_store"]["collection"]
            self.adb_metrics = config["index"]["vector_store"].get("metrics", "cosine")

        elif self.vectordb_type == "Hologres":
            self.hologres_host = config["index"]["vector_store"]["host"]
            self.hologres_port = config["index"]["vector_store"]["port"]
            self.hologres_user = config["index"]["vector_store"]["user"]
            self.hologres_password = config["index"]["vector_store"]["password"]
            self.hologres_database = config["index"]["vector_store"]["database"]
            self.hologres_table = config["index"]["vector_store"]["table_name"]
            self.hologres_pre_delete = config["index"]["vector_store"].get(
                "pre_delete_table", False
            )

        elif self.vectordb_type == "ElasticSearch":
            self.es_index = config["index"]["vector_store"]["es_index"]
            self.es_url = config["index"]["vector_store"]["es_url"]
            self.es_user = config["index"]["vector_store"]["es_user"]
            self.es_password = config["index"]["vector_store"]["es_password"]

        elif self.vectordb_type == "Milvus":
            self.milvus_host = config["index"]["vector_store"]["host"]
            self.milvus_port = config["index"]["vector_store"]["port"]
            self.milvus_user = config["index"]["vector_store"]["user"]
            self.milvus_password = config["index"]["vector_store"]["password"]
            self.milvus_database = config["index"]["vector_store"]["database"]
            self.milvus_collection_name = config["index"]["vector_store"][
                "collection_name"
            ]

        self.parser_type = config["node_parser"]["type"]
        self.chunk_size = config["node_parser"]["chunk_size"]
        self.chunk_overlap = config["node_parser"]["chunk_overlap"]

        self.reader_type = config["data_reader"].get("type", self.reader_type)
        self.enable_qa_extraction = config["data_reader"].get(
            "enable_qa_extraction", self.enable_qa_extraction
        )

        self.similarity_top_k = config["retriever"].get("similarity_top_k", 5)
        if config["retriever"]["retrieval_mode"] == "hybrid":
            self.retrieval_mode = "Hybrid"
            self.BM25_weight = config["retriever"]["BM25_weight"]
            self.vector_weight = config["retriever"]["vector_weight"]
            self.fusion_mode = config["retriever"]["fusion_mode"]
            self.query_rewrite_n = config["retriever"]["query_rewrite_n"]
        elif config["retriever"]["retrieval_mode"] == "embedding":
            self.retrieval_mode = "Embedding Only"
        elif config["retriever"]["retrieval_mode"] == "keyword":
            self.retrieval_mode = "Keyword Only"

        # if "Similarity" in config["postprocessor"]:
        #     self.similarity_cutoff = config["postprocessor"].get("similarity_cutoff", 0.1)

        rerank_model = config["postprocessor"].get("rerank_model", "no-reranker")
        if rerank_model == "llm-reranker":
            self.rerank_model = "llm-reranker"
        elif rerank_model == "bge-reranker-base":
            self.rerank_model = "bge-reranker-base"
        elif rerank_model == "bge-reranker-large":
            self.rerank_model = "bge-reranker-large"
        else:
            self.rerank_model = "no-reranker"

        self.synthesizer_type = config["synthesizer"].get("type", "SimpleSummarize")
        self.text_qa_template = config["synthesizer"].get("text_qa_template", None)

        if config["query_engine"]["type"] == "TransformQueryEngine":
            self.query_engine_type = "TransformQueryEngine"
        else:
            self.query_engine_type = "RetrieverQueryEngine"

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
        if self.llm == "PaiEas":
            config["llm"]["name"] = self.llm_eas_model_name
        else:
            config["llm"]["name"] = self.llm_api_model_name

        config["index"]["vector_store"]["type"] = self.vectordb_type
        config["index"]["persist_path"] = self.faiss_path

        config["node_parser"]["type"] = self.parser_type
        config["node_parser"]["chunk_size"] = int(self.chunk_size)
        config["node_parser"]["chunk_overlap"] = int(self.chunk_overlap)

        config["data_reader"]["enable_qa_extraction"] = self.enable_qa_extraction
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
            config["retriever"]["vector_weight"] = self.vector_weight
            config["retriever"]["BM25_weight"] = self.BM25_weight
            config["retriever"]["fusion_mode"] = self.fusion_mode
            config["retriever"]["query_rewrite_n"] = self.query_rewrite_n
        elif self.retrieval_mode == "Embedding Only":
            config["retriever"]["retrieval_mode"] = "embedding"
        elif self.retrieval_mode == "Keyword Only":
            config["retriever"]["retrieval_mode"] = "keyword"

        # config["postprocessor"]["similarity_cutoff"] = self.similarity_cutoff
        if self.rerank_model == "llm-reranker":
            config["postprocessor"]["rerank_model"] = "llm-reranker"
        elif self.rerank_model == "bge-reranker-base":
            config["postprocessor"]["rerank_model"] = "bge-reranker-base"
        elif self.rerank_model == "bge-reranker-large":
            config["postprocessor"]["rerank_model"] = "bge-reranker-large"
        else:
            config["postprocessor"]["rerank_model"] = "no-reranker"
        config["postprocessor"]["top_n"] = self.similarity_top_k

        config["synthesizer"]["type"] = self.synthesizer_type
        config["synthesizer"]["text_qa_template"] = self.text_qa_template
        if self.query_engine_type == "TransformQueryEngine":
            config["query_engine"]["type"] = "TransformQueryEngine"
        else:
            config["query_engine"]["type"] = "RetrieverQueryEngine"

        return _transform_to_dict(config)

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
        }
        settings["chunk_size"] = {"value": self.chunk_size}
        settings["chunk_overlap"] = {"value": self.chunk_overlap}
        settings["enable_qa_extraction"] = {"value": self.enable_qa_extraction}
        settings["similarity_top_k"] = {"value": self.similarity_top_k}
        settings["rerank_model"] = {"value": self.rerank_model}
        settings["retrieval_mode"] = {"value": self.retrieval_mode}

        prm_type = PROMPT_MAP.get(view_model.text_qa_template, "Custom")
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

        return settings


view_model = ViewModel()
