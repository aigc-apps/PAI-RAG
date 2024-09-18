from pydantic import BaseModel
from typing import Any, Dict
from collections import defaultdict
from pai_rag.app.web.ui_constants import (
    EMBEDDING_DIM_DICT,
    DEFAULT_EMBED_SIZE,
    DEFAULT_HF_EMBED_MODEL,
    LLM_MODEL_KEY_DICT,
    MLLM_MODEL_KEY_DICT,
    PROMPT_MAP,
)
import pandas as pd
import os
import re
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
    llm_eas_model_name: str = "model_name"
    llm_api_key: str = None
    llm_api_model_name: str = None
    llm_temperature: float = 0.1

    # mllm
    use_mllm: bool = False
    mllm: str = None
    mllm_eas_url: str = None
    mllm_eas_token: str = None
    mllm_eas_model_name: str = "model_name"
    mllm_api_key: str = None
    mllm_api_model_name: str = None

    # oss
    use_oss: bool = False
    oss_ak: str = None
    oss_sk: str = None
    oss_endpoint: str = None
    oss_bucket: str = None
    oss_prefix: str = None

    # chunking
    parser_type: str = "Sentence"
    chunk_size: int = 500
    chunk_overlap: int = 20

    # reader
    reader_type: str = "SimpleDirectoryReader"
    enable_qa_extraction: bool = False
    enable_raptor: bool = False
    enable_multimodal: bool = False
    enable_table_summary: bool = False

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
    milvus_database: str = "default"
    milvus_collection_name: str = "pairagcollection"

    # open search
    opensearch_endpoint: str = None
    opensearch_instance_id: str = None
    opensearch_username: str = None
    opensearch_password: str = None
    opensearch_table_name: str = "pairag"

    # PostgreSQL
    postgresql_host: str = None
    postgresql_port: int = 5432
    postgresql_database: str = None
    postgresql_table_name: str = "pairag"
    postgresql_username: str = None
    postgresql_password: str = None

    # retriever
    similarity_top_k: int = 5
    image_similarity_top_k: int = 2
    need_image: bool = False
    retrieval_mode: str = "hybrid"  # hybrid / embedding / keyword
    query_rewrite_n: int = 1

    # websearch
    search_api_key: str = None
    search_count: int = 10
    search_lang: str = "zh-CN"

    # data_analysis
    analysis_type: str = "nl2pandas"  # nl2sql / nl2pandas
    analysis_file_path: str = None
    db_dialect: str = "mysql"
    db_username: str = None
    db_password: str = None
    db_host: str = None
    db_port: int = 3306
    db_name: str = None
    db_tables: str = None
    db_descriptions: str = None
    db_nl2sql_prompt: str = None

    # postprocessor
    reranker_type: str = (
        "simple-weighted-reranker"  # simple-weighted-reranker / model-based-reranker
    )
    reranker_model: str = "bge-reranker-base"  # bge-reranker-base / bge-reranker-large
    keyword_weight: float = 0.3
    vector_weight: float = 0.7
    similarity_threshold: float = None

    query_engine_type: str = None

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

        view_model.use_mllm = config["llm"]["multi_modal"].get(
            "enable", view_model.use_mllm
        )
        view_model.mllm = config["llm"]["multi_modal"].get("source", view_model.mllm)
        view_model.mllm_eas_url = config["llm"]["multi_modal"].get(
            "endpoint", view_model.mllm_eas_url
        )
        view_model.mllm_eas_token = config["llm"]["multi_modal"].get(
            "token", view_model.mllm_eas_token
        )
        view_model.mllm_api_key = config["llm"]["multi_modal"].get(
            "api_key", view_model.mllm_api_key
        )

        if view_model.mllm.lower() == "paieas":
            print(
                "view_model.mllm_eas_model_name",
                view_model.mllm_eas_model_name,
                "2",
                config["llm"]["multi_modal"]["name"],
            )
            view_model.mllm_eas_model_name = config["llm"]["multi_modal"].get(
                "name", view_model.mllm_eas_model_name
            )
        else:
            view_model.mllm_api_model_name = config["llm"]["multi_modal"].get(
                "name", view_model.mllm_api_model_name
            )

        view_model.use_oss = config["oss_store"].get("enable", view_model.use_oss)
        view_model.oss_ak = config["oss_store"].get("ak", view_model.oss_ak)
        view_model.oss_sk = config["oss_store"].get("sk", view_model.oss_sk)
        view_model.oss_endpoint = config["oss_store"].get(
            "endpoint", view_model.oss_endpoint
        )
        view_model.oss_bucket = config["oss_store"].get("bucket", view_model.oss_bucket)
        view_model.oss_prefix = config["oss_store"].get("prefix", view_model.oss_prefix)

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

        elif view_model.vectordb_type.lower() == "opensearch":
            view_model.opensearch_endpoint = config["index"]["vector_store"]["endpoint"]
            view_model.opensearch_instance_id = config["index"]["vector_store"][
                "instance_id"
            ]
            view_model.opensearch_username = config["index"]["vector_store"]["username"]
            view_model.opensearch_password = config["index"]["vector_store"]["password"]
            view_model.opensearch_table_name = config["index"]["vector_store"][
                "table_name"
            ]

        elif view_model.vectordb_type.lower() == "postgresql":
            view_model.postgresql_host = config["index"]["vector_store"]["host"]
            view_model.postgresql_port = config["index"]["vector_store"]["port"]
            view_model.postgresql_database = config["index"]["vector_store"]["database"]
            view_model.postgresql_table_name = config["index"]["vector_store"][
                "table_name"
            ]
            view_model.postgresql_username = config["index"]["vector_store"]["username"]
            view_model.postgresql_password = config["index"]["vector_store"]["password"]

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
        view_model.enable_multimodal = config["data_reader"].get(
            "enable_multimodal", view_model.enable_multimodal
        )
        view_model.enable_table_summary = config["data_reader"].get(
            "enable_table_summary", view_model.enable_table_summary
        )

        view_model.similarity_top_k = config["retriever"].get("similarity_top_k", 5)
        view_model.image_similarity_top_k = config["retriever"].get(
            "image_similarity_top_k", 2
        )
        view_model.need_image = config["retriever"].get("need_image", False)
        if config["retriever"]["retrieval_mode"] == "hybrid":
            view_model.retrieval_mode = "Hybrid"
            view_model.query_rewrite_n = config["retriever"]["query_rewrite_n"]
        elif config["retriever"]["retrieval_mode"] == "embedding":
            view_model.retrieval_mode = "Embedding Only"
        elif config["retriever"]["retrieval_mode"] == "keyword":
            view_model.retrieval_mode = "Keyword Only"

        if config["data_analysis"]["analysis_type"] == "nl2pandas":
            view_model.analysis_type = "nl2pandas"
        elif config["data_analysis"]["analysis_type"] == "nl2sql":
            view_model.analysis_type = "nl2sql"

        view_model.analysis_file_path = config["data_analysis"].get(
            "analysis_file_path", None
        )
        view_model.db_dialect = config["data_analysis"].get("dialect", "mysql")
        view_model.db_username = config["data_analysis"].get("user", None)
        view_model.db_password = config["data_analysis"].get("password", None)
        view_model.db_host = config["data_analysis"].get("host", None)
        view_model.db_port = config["data_analysis"].get("port", 3306)
        view_model.db_name = config["data_analysis"].get("dbname", None)

        # from list to string
        if config["data_analysis"].get("tables", None):
            view_model.db_tables = ",".join(config["data_analysis"].get("tables", None))
        else:
            view_model.db_tables = None

        # from dict to string
        if config["data_analysis"].get("descriptions", None):
            view_model.db_descriptions = json.dumps(
                config["data_analysis"].get("descriptions", None), ensure_ascii=False
            )
        else:
            view_model.db_descriptions = None

        view_model.db_nl2sql_prompt = config["data_analysis"].get("nl2sql_prompt", None)

        reranker_type = config["postprocessor"].get(
            "reranker_type", "simple-weighted-reranker"
        )
        similarity_threshold = config["postprocessor"].get("similarity_threshold", 0)
        view_model.similarity_threshold = (
            similarity_threshold
            if similarity_threshold and similarity_threshold > 0
            else None
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

        search_config = config.get("search") or {}
        view_model.search_api_key = search_config.get(
            "search_api_key"
        ) or os.environ.get("BING_SEARCH_KEY")
        view_model.search_lang = search_config.get("search_lang", "zh-CN")
        view_model.search_count = search_config.get("search_count", 10)

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

        config["llm"]["multi_modal"]["enable"] = self.use_mllm
        config["llm"]["multi_modal"]["source"] = self.mllm
        config["llm"]["multi_modal"]["endpoint"] = self.mllm_eas_url
        config["llm"]["multi_modal"]["token"] = self.mllm_eas_token
        config["llm"]["multi_modal"]["api_key"] = self.mllm_api_key
        if self.mllm.lower() == "paieas":
            config["llm"]["multi_modal"]["name"] = self.mllm_eas_model_name
        else:
            config["llm"]["multi_modal"]["name"] = self.mllm_api_model_name

        config["oss_store"]["enable"] = self.use_oss
        if os.getenv("OSS_ACCESS_KEY_ID") is None and self.oss_ak:
            os.environ["OSS_ACCESS_KEY_ID"] = self.oss_ak
        if os.getenv("OSS_ACCESS_KEY_SECRET") is None and self.oss_sk:
            os.environ["OSS_ACCESS_KEY_SECRET"] = self.oss_sk
        if self.oss_ak and "***" not in self.oss_ak:
            config["oss_store"]["ak"] = self.oss_ak
        if self.oss_sk and "***" not in self.oss_sk:
            config["oss_store"]["sk"] = self.oss_sk
        config["oss_store"]["endpoint"] = self.oss_endpoint
        config["oss_store"]["bucket"] = self.oss_bucket
        config["oss_store"]["prefix"] = self.oss_prefix

        config["index"]["vector_store"]["type"] = self.vectordb_type
        config["index"]["persist_path"] = self.faiss_path

        config["node_parser"]["type"] = self.parser_type
        config["node_parser"]["chunk_size"] = int(self.chunk_size)
        config["node_parser"]["chunk_overlap"] = int(self.chunk_overlap)

        config["data_reader"]["enable_qa_extraction"] = self.enable_qa_extraction
        config["data_reader"]["enable_raptor"] = self.enable_raptor
        config["data_reader"]["enable_multimodal"] = self.enable_multimodal
        config["data_reader"]["enable_table_summary"] = self.enable_table_summary
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

        elif self.vectordb_type.lower() == "opensearch":
            config["index"]["vector_store"]["endpoint"] = self.opensearch_endpoint
            config["index"]["vector_store"]["instance_id"] = self.opensearch_instance_id
            config["index"]["vector_store"]["username"] = self.opensearch_username
            config["index"]["vector_store"]["password"] = self.opensearch_password
            config["index"]["vector_store"]["table_name"] = self.opensearch_table_name

        elif self.vectordb_type.lower() == "postgresql":
            config["index"]["vector_store"]["host"] = self.postgresql_host
            config["index"]["vector_store"]["port"] = self.postgresql_port
            config["index"]["vector_store"]["database"] = self.postgresql_database
            config["index"]["vector_store"]["table_name"] = self.postgresql_table_name
            config["index"]["vector_store"]["username"] = self.postgresql_username
            config["index"]["vector_store"]["password"] = self.postgresql_password

        config["retriever"]["similarity_top_k"] = self.similarity_top_k
        config["retriever"]["image_similarity_top_k"] = self.image_similarity_top_k
        config["retriever"]["need_image"] = self.need_image
        if self.retrieval_mode == "Hybrid":
            config["retriever"]["retrieval_mode"] = "hybrid"
            config["retriever"]["query_rewrite_n"] = self.query_rewrite_n
        elif self.retrieval_mode == "Embedding Only":
            config["retriever"]["retrieval_mode"] = "embedding"
        elif self.retrieval_mode == "Keyword Only":
            config["retriever"]["retrieval_mode"] = "keyword"

        if self.analysis_type == "nl2pandas":
            config["data_analysis"]["analysis_type"] = "nl2pandas"
        elif self.analysis_type == "nl2sql":
            config["data_analysis"]["analysis_type"] = "nl2sql"

        config["data_analysis"]["analysis_file_path"] = self.analysis_file_path
        config["data_analysis"]["dialect"] = self.db_dialect
        config["data_analysis"]["user"] = self.db_username
        config["data_analysis"]["password"] = self.db_password
        config["data_analysis"]["host"] = self.db_host
        config["data_analysis"]["port"] = self.db_port
        config["data_analysis"]["dbname"] = self.db_name
        config["data_analysis"]["nl2sql_prompt"] = self.db_nl2sql_prompt
        # string to list
        if self.db_tables:
            # 去掉首位空格和末尾逗号
            value = self.db_tables.strip().rstrip(",")
            # 英文逗号和中文逗号作为分隔符进行分割，并去除多余空白字符
            value = [word.strip() for word in re.split(r"\s*,\s*|，\s*", value)]
            config["data_analysis"]["tables"] = value
        else:
            config["data_analysis"]["tables"] = None
        # string to dict
        if self.db_descriptions:
            config["data_analysis"]["descriptions"] = json.loads(self.db_descriptions)
        else:
            config["data_analysis"]["descriptions"] = None

        config["postprocessor"]["reranker_type"] = self.reranker_type
        config["postprocessor"]["reranker_model"] = self.reranker_model
        config["postprocessor"]["keyword_weight"] = self.keyword_weight
        config["postprocessor"]["vector_weight"] = self.vector_weight
        config["postprocessor"]["similarity_threshold"] = self.similarity_threshold
        config["postprocessor"]["top_n"] = self.similarity_top_k

        config["synthesizer"]["type"] = self.synthesizer_type
        config["synthesizer"]["text_qa_template"] = self.text_qa_template

        config["search"]["search_api_key"] = self.search_api_key or os.environ.get(
            "BING_SEARCH_KEY"
        )
        config["search"]["search_lang"] = self.search_lang
        config["search"]["search_count"] = self.search_count

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
        settings["llm_eas_url"] = {
            "value": self.llm_eas_url,
            "visible": self.llm.lower() == "paieas",
        }
        settings["llm_eas_token"] = {
            "value": self.llm_eas_token,
            "visible": self.llm.lower() == "paieas",
        }
        if self.llm.lower() == "paieas" and not self.llm_eas_model_name:
            self.llm_eas_model_name = "model_name"

        settings["llm_eas_model_name"] = {
            "value": self.llm_eas_model_name,
            "visible": self.llm.lower() == "paieas",
        }
        settings["llm_api_model_name"] = {
            "value": self.llm_api_model_name,
            "choices": LLM_MODEL_KEY_DICT.get(self.llm, []),
            "visible": self.llm.lower() != "paieas",
        }

        settings["use_mllm"] = {"value": self.use_mllm}
        settings["mllm"] = {"value": self.mllm}
        settings["mllm_eas_url"] = {"value": self.mllm_eas_url}
        settings["mllm_eas_token"] = {"value": self.mllm_eas_token}
        settings["mllm_eas_model_name"] = {"value": self.mllm_eas_model_name}
        settings["mllm_api_model_name"] = {
            "value": self.mllm_api_model_name,
            "choices": MLLM_MODEL_KEY_DICT.get(self.mllm, []),
            "visible": self.mllm.lower() != "paieas",
        }

        settings["use_oss"] = {"value": self.use_oss}
        settings["oss_ak"] = {
            "value": (self.oss_ak[:2] + "*" * (len(self.oss_ak) - 4) + self.oss_ak[-2:])
            if self.oss_ak
            else self.oss_ak,
            "type": "text" if self.oss_ak else "password",
        }
        settings["oss_sk"] = {
            "value": (self.oss_sk[:2] + "*" * (len(self.oss_sk) - 4) + self.oss_sk[-2:])
            if self.oss_sk
            else self.oss_sk,
            "type": "text" if self.oss_sk else "password",
        }
        settings["oss_endpoint"] = {"value": self.oss_endpoint}
        settings["oss_bucket"] = {"value": self.oss_bucket}
        settings["oss_prefix"] = {"value": self.oss_prefix}

        settings["chunk_size"] = {"value": self.chunk_size}
        settings["chunk_overlap"] = {"value": self.chunk_overlap}
        settings["enable_qa_extraction"] = {"value": self.enable_qa_extraction}
        settings["enable_raptor"] = {"value": self.enable_raptor}
        settings["enable_multimodal"] = {"value": self.enable_multimodal}
        settings["enable_table_summary"] = {"value": self.enable_table_summary}

        # retrieval and rerank
        settings["retrieval_mode"] = {"value": self.retrieval_mode}
        settings["reranker_type"] = {"value": self.reranker_type}
        settings["similarity_top_k"] = {"value": self.similarity_top_k}
        settings["image_similarity_top_k"] = {"value": self.image_similarity_top_k}
        settings["need_image"] = {"value": self.need_image}
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

        # opensearch
        settings["opensearch_endpoint"] = {"value": self.opensearch_endpoint}
        settings["opensearch_instance_id"] = {"value": self.opensearch_instance_id}
        settings["opensearch_username"] = {"value": self.opensearch_username}
        settings["opensearch_password"] = {"value": self.opensearch_password}
        settings["opensearch_table_name"] = {"value": self.opensearch_table_name}

        # postgresql
        settings["postgresql_host"] = {"value": self.postgresql_host}
        settings["postgresql_port"] = {"value": self.postgresql_port}
        settings["postgresql_database"] = {"value": self.postgresql_database}
        settings["postgresql_table_name"] = {"value": self.postgresql_table_name}
        settings["postgresql_username"] = {"value": self.postgresql_username}
        settings["postgresql_password"] = {"value": self.postgresql_password}

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

        # search
        settings["search_api_key"] = {"value": self.search_api_key}
        settings["search_lang"] = {"value": self.search_lang}
        settings["search_count"] = {"value": self.search_count}

        # data_analysis
        settings["analysis_type"] = {"value": self.analysis_type}
        settings["analysis_file_path"] = {"value": self.analysis_file_path}
        settings["db_dialect"] = {"value": self.db_dialect}
        settings["db_username"] = {"value": self.db_username}
        settings["db_password"] = {"value": self.db_password}
        settings["db_host"] = {"value": self.db_host}
        settings["db_port"] = {"value": self.db_port}
        settings["db_name"] = {"value": self.db_name}
        settings["db_tables"] = {"value": self.db_tables}
        settings["db_descriptions"] = {"value": self.db_descriptions}
        settings["db_nl2sql_prompt"] = {"value": self.db_nl2sql_prompt}

        return settings
