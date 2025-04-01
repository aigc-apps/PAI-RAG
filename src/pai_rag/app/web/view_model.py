from pydantic import BaseModel
from typing import Any, Dict, List
from collections import defaultdict
import pandas as pd
import os
import re
from datetime import datetime
import tempfile
import json

from llama_index.core.vector_stores.types import VectorStoreQueryMode
from pai_rag.core.rag_config import RagConfig
from pai_rag.integrations.data_analysis.data_analysis_config import (
    MysqlAnalysisConfig,
    PandasAnalysisConfig,
    SqliteAnalysisConfig,
)
from pai_rag.integrations.llms.pai.llm_config import (
    PaiBaseLlmConfig,
)
from pai_rag.integrations.postprocessor.pai.pai_postprocessor import (
    SimilarityPostProcessorConfig,
)
from pai_rag.integrations.search.search_config import (
    DEFAULT_ALIYUN_SEARCH_ENDPOINT,
    DEFAULT_SEARCH_COUNT,
    BingSearchConfig,
    QuarkSearchConfig,
    AliyunSearchConfig,
    GoogleSearchConfig,
)
from pai_rag.integrations.postprocessor.pai.pai_postprocessor import PostProcessorType


QUERY_TYPE_MAP = {
    "对话 (大模型)": "llm",
    "检索测试": "retrieval",
    "对话 (网络搜索)": "websearch",
    "对话 (知识库)": "rag",
}
INVERTED_QUERY_TYPE_MAP = {
    "llm": "对话 (大模型)",
    "retrieval": "检索测试",
    "websearch": "对话 (网络搜索)",
    "rag": "对话 (知识库)",
}


def recursive_dict():
    return defaultdict(recursive_dict)


def _transform_to_dict(config):
    for k, v in config.items():
        if isinstance(v, defaultdict):
            config[k] = _transform_to_dict(v)
    return dict(config)


class ViewModel(BaseModel):
    chat_model_id: str = "default"

    query_rewrite_model_id: str = "default"

    data_analysis_model_id: str = "default"

    # oss
    use_oss: bool = False
    oss_ak: str = None
    oss_sk: str = None
    oss_endpoint: str = None
    oss_bucket: str = None

    # node_parser
    parser_type: str = "Sentence"
    chunk_size: int = 500
    chunk_overlap: int = 20
    enable_multimodal: bool = False

    # reader
    reader_type: str = "SimpleDirectoryReader"
    enable_mandatory_ocr: bool = False
    number_workers: int = 4

    config_file: str = None

    # retriever
    similarity_top_k: int = 5
    image_similarity_top_k: int = 2
    need_image: bool = False
    retrieval_mode: str = "混合检索"  # "向量检索" / "关键字检索" / "混合检索"
    query_rewrite_n: int = 1

    # websearch
    default_web_search: bool = False
    search_type: str = "bing"
    search_api_key: str = None
    search_count: int = DEFAULT_SEARCH_COUNT
    search_lang: str = "zh-CN"

    aliyun_endpoint: str = DEFAULT_ALIYUN_SEARCH_ENDPOINT
    aliyun_access_key_id: str = None
    aliyun_access_key_secret: str = None

    serpapi_key: str = None

    # data_analysis
    analysis_type: str = "nl2pandas"  # nl2sql / nl2pandas
    analysis_file_path: str = None
    db_dialect: str = "mysql"
    db_username: str = None
    db_password: str = None
    db_host: str = None
    db_port: int = 3306
    database: str = None
    db_tables: str = None
    db_descriptions: str = None
    enable_enhanced_description: bool = False
    enable_db_history: bool = False
    enable_db_embedding: bool = False
    max_col_num: int = 100
    max_val_num: int = 10000
    enable_query_preprocessor: bool = False
    enable_db_preretriever: bool = False
    enable_db_selector: bool = False
    db_nl2sql_prompt: str = None
    synthesizer_prompt: str = None
    # da_llm_base_url: str = None
    # da_llm_api_key: str = None
    # da_llm_model_name: str = "default"
    # da_llm_max_tokens: int = 1024

    # postprocessor
    reranker_type: str = "无重排序"  # 无重排序 / 基于模型的重排序
    reranker_model: str = "bge-reranker-base"  # bge-reranker-base / bge-reranker-large
    keyword_weight: float = 0.3
    vector_weight: float = 0.7
    similarity_threshold: float = 0.5
    reranker_similarity_threshold: float = 0
    reranker_similarity_top_k: int = 3

    query_type: str = "对话 (知识库)"

    enable_query_transform: bool = True
    rewrite_base_prompt: str = None
    rewrite_llm_prompt: str = None
    rewrite_knowledgebase_prompt: str = None
    rewrite_agent_prompt: str = None
    rewrite_search_prompt: str = None
    rewrite_db_prompt: str = None

    synthesizer_type: str = None
    system_role_template: str = None
    custom_prompt_template: str = None
    # multimodal_qa_template: str = None
    # citation_multimodal_qa_template: str = None

    # agent
    agent_api_definition: str = None  # API tool definition
    agent_function_definition: str = None  # Function tool definition
    agent_python_scripts: str = None  # Function scripts
    agent_system_prompt: str = None  # Agent system prompt

    # intent
    intent_description: str = None

    # guardrail
    guardrail_ak: str = None
    guardrail_sk: str = None
    guardrail_endpoint: str = None
    guardrail_region: str = None
    enable_guardrail: bool = False

    # llms
    llms: List[PaiBaseLlmConfig] = None

    def update(self, update_paras: Dict[str, Any]):
        attr_set = set(dir(self))
        for key, value in update_paras.items():
            if key in attr_set:
                setattr(self, key, value)

    @staticmethod
    def from_app_config(config: RagConfig):
        view_model = ViewModel()

        view_model.default_web_search = config.system.default_web_search

        view_model.llms = config.llms

        view_model.chat_model_id = config.chat.model_id or "default"

        view_model.query_type = INVERTED_QUERY_TYPE_MAP.get(
            config.system.query_type, "对话 (知识库)"
        )

        view_model.use_oss = (
            config.oss_store.bucket is not None and config.oss_store.bucket != ""
        )
        view_model.oss_ak = config.oss_store.ak
        view_model.oss_sk = config.oss_store.sk
        view_model.oss_endpoint = config.oss_store.endpoint
        view_model.oss_bucket = config.oss_store.bucket

        view_model.parser_type = config.node_parser.type
        view_model.chunk_overlap = config.node_parser.chunk_overlap
        view_model.chunk_size = config.node_parser.chunk_size

        view_model.enable_mandatory_ocr = config.data_reader.enable_mandatory_ocr
        view_model.number_workers = config.data_reader.number_workers

        view_model.similarity_top_k = config.retriever.similarity_top_k
        view_model.image_similarity_top_k = config.retriever.image_similarity_top_k
        view_model.need_image = config.retriever.search_image
        view_model.vector_weight = config.retriever.hybrid_fusion_weights[0]
        view_model.keyword_weight = config.retriever.hybrid_fusion_weights[1]
        if config.retriever.vector_store_query_mode == VectorStoreQueryMode.DEFAULT:
            view_model.retrieval_mode = "向量检索"
        elif config.retriever.vector_store_query_mode == VectorStoreQueryMode.HYBRID:
            view_model.retrieval_mode = "混合检索"
        else:
            view_model.retrieval_mode = "关键字检索"

        if config.postprocessor.reranker_type.value == PostProcessorType.reranker_model:
            view_model.reranker_type = "基于模型的重排序"
        else:
            view_model.reranker_type = "无重排序"

        if isinstance(config.postprocessor, SimilarityPostProcessorConfig):
            view_model.similarity_threshold = config.postprocessor.similarity_threshold
        else:
            view_model.reranker_model = config.postprocessor.reranker_model
            view_model.reranker_similarity_top_k = config.postprocessor.top_n
            view_model.reranker_similarity_threshold = (
                config.postprocessor.similarity_threshold
            )

        view_model.enable_query_transform = config.query_rewrite.enabled
        view_model.rewrite_base_prompt = config.query_rewrite.base_prompt_template_str
        view_model.rewrite_llm_prompt = config.query_rewrite.llm_tool_prompt_str
        view_model.rewrite_knowledgebase_prompt = (
            config.query_rewrite.knowledge_tool_prompt_str
        )

        view_model.rewrite_agent_prompt = config.query_rewrite.agent_tool_prompt_str
        view_model.rewrite_search_prompt = (
            config.query_rewrite.websearch_tool_prompt_str
        )
        view_model.rewrite_db_prompt = config.query_rewrite.db_tool_prompt_str

        view_model.query_rewrite_model_id = config.query_rewrite.model_id or "default"

        view_model.system_role_template = config.synthesizer.system_role_template
        view_model.custom_prompt_template = config.synthesizer.custom_prompt_template
        # view_model.multimodal_qa_template = config.synthesizer.multimodal_qa_template
        # view_model.citation_multimodal_qa_template = (
        #     config.synthesizer.citation_multimodal_qa_template
        # )

        if isinstance(config.search, BingSearchConfig):
            view_model.search_type = "bing"
            view_model.search_api_key = config.search.search_api_key or os.environ.get(
                "BING_SEARCH_KEY"
            )
            view_model.search_lang = config.search.search_lang
            view_model.search_count = config.search.search_count
        elif isinstance(config.search, QuarkSearchConfig):
            view_model.search_type = "aliyun"
            view_model.aliyun_endpoint = ""
            view_model.aliyun_access_key_id = ""
            view_model.aliyun_access_key_secret = ""
            view_model.search_count = config.search.search_count
        elif isinstance(config.search, AliyunSearchConfig):
            view_model.search_type = "aliyun"
            view_model.aliyun_endpoint = config.search.endpoint
            view_model.aliyun_access_key_id = config.search.access_key_id
            view_model.aliyun_access_key_secret = config.search.access_key_secret
            view_model.search_count = config.search.search_count
        elif isinstance(config.search, GoogleSearchConfig):
            view_model.search_type = "google"
            view_model.serpapi_key = config.search.serpapi_key or os.environ.get(
                "SERPAPI_KEY"
            )
            view_model.search_lang = config.search.search_lang
            view_model.search_count = config.search.search_count

        view_model.data_analysis_model_id = config.data_analysis.model_id or "default"

        if isinstance(config.data_analysis, PandasAnalysisConfig):
            view_model.analysis_type = "nl2pandas"
            view_model.analysis_file_path = config.data_analysis.file_path

        elif isinstance(config.data_analysis, SqliteAnalysisConfig):
            view_model.analysis_type = "nl2sql"
            view_model.db_dialect = config.data_analysis.type.value
            view_model.database = config.data_analysis.database
        elif isinstance(config.data_analysis, MysqlAnalysisConfig):
            view_model.analysis_type = "nl2sql"
            view_model.db_dialect = config.data_analysis.type.value
            view_model.database = config.data_analysis.database
            view_model.db_username = config.data_analysis.user
            view_model.db_password = config.data_analysis.password
            view_model.db_host = config.data_analysis.host
            view_model.db_port = config.data_analysis.port
            view_model.db_tables = ",".join(config.data_analysis.tables)
            view_model.db_descriptions = (
                json.dumps(config.data_analysis.descriptions, ensure_ascii=False)
                if config.data_analysis.descriptions
                else None
            )
            view_model.enable_enhanced_description = (
                config.data_analysis.enable_enhanced_description
            )
            view_model.enable_db_history = config.data_analysis.enable_db_history
            view_model.enable_db_embedding = config.data_analysis.enable_db_embedding
            view_model.max_col_num = config.data_analysis.max_col_num
            view_model.max_val_num = config.data_analysis.max_val_num
            view_model.enable_query_preprocessor = (
                config.data_analysis.enable_query_preprocessor
            )
            view_model.enable_db_preretriever = (
                config.data_analysis.enable_db_preretriever
            )
            view_model.enable_db_selector = config.data_analysis.enable_db_selector
        view_model.db_nl2sql_prompt = config.data_analysis.nl2sql_prompt
        view_model.synthesizer_prompt = config.data_analysis.synthesizer_prompt

        # if config.data_analysis.llm is not None:
        #     view_model.da_llm_base_url = config.data_analysis.llm.base_url
        #     view_model.da_llm_api_key = config.data_analysis.llm.api_key
        #     view_model.da_llm_model_name = config.data_analysis.llm.model

        view_model.agent_api_definition = config.agent.api_definition
        view_model.agent_function_definition = config.agent.function_definition
        view_model.agent_python_scripts = config.agent.python_scripts
        view_model.agent_system_prompt = config.agent.system_prompt

        view_model.intent_description = json.dumps(
            config.intent.descriptions, ensure_ascii=False, sort_keys=True, indent=4
        )

        if config.guardrail.is_enabled():
            view_model.enable_guardrail = True
            view_model.guardrail_ak = config.guardrail.access_key_id
            view_model.guardrail_sk = config.guardrail.access_key_secret
            view_model.guardrail_endpoint = config.guardrail.endpoint
            view_model.guardrail_region = config.guardrail.region

        return view_model

    def to_app_config(self):
        config = recursive_dict()

        config["system"]["default_web_search"] = self.default_web_search

        config["system"]["query_type"] = QUERY_TYPE_MAP.get(self.query_type, "rag")

        config["chat"]["model_id"] = self.chat_model_id
        config["query_rewrite"]["model_id"] = self.query_rewrite_model_id
        config["data_analysis"]["model_id"] = self.data_analysis_model_id

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

        config["node_parser"]["type"] = self.parser_type
        config["node_parser"]["chunk_size"] = int(self.chunk_size)
        config["node_parser"]["chunk_overlap"] = int(self.chunk_overlap)

        config["data_reader"]["enable_mandatory_ocr"] = self.enable_mandatory_ocr
        config["data_reader"]["number_workers"] = int(self.number_workers)

        config["retriever"]["similarity_top_k"] = self.similarity_top_k
        config["retriever"]["image_similarity_top_k"] = self.image_similarity_top_k
        config["retriever"]["vector_weight"] = self.vector_weight
        config["retriever"]["keyword_weight"] = self.keyword_weight

        config["retriever"]["search_image"] = self.need_image
        if self.retrieval_mode == "混合检索":
            config["retriever"]["vector_store_query_mode"] = VectorStoreQueryMode.HYBRID
            config["retriever"]["hybrid_fusion_weights"] = [
                self.vector_weight,
                self.keyword_weight,
            ]
        elif self.retrieval_mode == "向量检索":
            config["retriever"][
                "vector_store_query_mode"
            ] = VectorStoreQueryMode.DEFAULT
        elif self.retrieval_mode == "关键字检索":
            config["retriever"][
                "vector_store_query_mode"
            ] = VectorStoreQueryMode.TEXT_SEARCH

        if self.analysis_type == "nl2pandas":
            config["data_analysis"]["type"] = "pandas"
            config["data_analysis"]["file_path"] = self.analysis_file_path
        elif self.analysis_type == "nl2sql":
            config["data_analysis"]["type"] = "mysql"
            config["data_analysis"]["user"] = self.db_username
            config["data_analysis"]["password"] = self.db_password
            config["data_analysis"]["host"] = self.db_host
            config["data_analysis"]["port"] = self.db_port
            config["data_analysis"]["database"] = self.database
            config["data_analysis"][
                "enable_enhanced_description"
            ] = self.enable_enhanced_description
            config["data_analysis"]["enable_db_embedding"] = self.enable_db_embedding
            config["data_analysis"]["max_col_num"] = self.max_col_num
            config["data_analysis"]["max_val_num"] = self.max_val_num
            config["data_analysis"]["enable_db_history"] = self.enable_db_history
            config["data_analysis"][
                "enable_query_preprocessor"
            ] = self.enable_query_preprocessor
            config["data_analysis"][
                "enable_db_preretriever"
            ] = self.enable_db_preretriever
            config["data_analysis"]["enable_db_selector"] = self.enable_db_selector
            config["data_analysis"]["nl2sql_prompt"] = self.db_nl2sql_prompt
            config["data_analysis"]["synthesizer_prompt"] = self.synthesizer_prompt

            # string to list
            if self.db_tables:
                # 去掉首位空格和末尾逗号
                value = self.db_tables.strip().rstrip(",")
                # 英文逗号和中文逗号作为分隔符进行分割，并去除多余空白字符
                value = [word.strip() for word in re.split(r"\s*,\s*|，\s*", value)]
                config["data_analysis"]["tables"] = value
            else:
                config["data_analysis"]["tables"] = []
            # string to dict
            if self.db_descriptions:
                config["data_analysis"]["descriptions"] = json.loads(
                    self.db_descriptions
                )
            else:
                config["data_analysis"]["descriptions"] = {}
        # config["data_analysis"]["llm"]["source"] = SupportedLlmType.openai_compatible
        # config["data_analysis"]["llm"]["base_url"] = self.da_llm_base_url
        # config["data_analysis"]["llm"]["api_key"] = self.da_llm_api_key
        # config["data_analysis"]["llm"]["model"] = self.da_llm_model_name
        # config["data_analysis"]["llm"]["max_tokens"] = self.da_llm_max_tokens

        if self.reranker_type == "基于模型的重排序":
            config["postprocessor"]["reranker_type"] = PostProcessorType.reranker_model
        else:
            config["postprocessor"]["reranker_type"] = PostProcessorType.no_reranker

        config["postprocessor"]["reranker_model"] = self.reranker_model
        if self.reranker_type == "无重排序":
            config["postprocessor"]["similarity_threshold"] = self.similarity_threshold
        else:
            config["postprocessor"][
                "similarity_threshold"
            ] = self.reranker_similarity_threshold
            config["postprocessor"]["top_n"] = self.reranker_similarity_top_k

        config["synthesizer"]["custom_prompt_template"] = self.custom_prompt_template
        config["synthesizer"]["system_role_template"] = self.system_role_template

        config["query_rewrite"]["base_prompt_template_str"] = self.rewrite_base_prompt
        config["query_rewrite"]["llm_tool_prompt_str"] = self.rewrite_llm_prompt
        config["query_rewrite"][
            "knowledge_tool_prompt_str"
        ] = self.rewrite_knowledgebase_prompt
        config["query_rewrite"]["agent_tool_prompt_str"] = self.rewrite_agent_prompt
        config["query_rewrite"][
            "websearch_tool_prompt_str"
        ] = self.rewrite_search_prompt
        config["query_rewrite"]["db_tool_prompt_str"] = self.rewrite_db_prompt

        config["query_rewrite"]["enabled"] = self.enable_query_transform
        # config["synthesizer"]["multimodal_qa_template"] = self.multimodal_qa_template
        # config["synthesizer"][
        #     "citation_multimodal_qa_template"
        # ] = self.citation_multimodal_qa_template

        if self.search_type == "bing":
            config["search"]["source"] = "bing"
            config["search"]["search_api_key"] = self.search_api_key or os.environ.get(
                "BING_SEARCH_KEY"
            )
            config["search"]["search_lang"] = self.search_lang
            config["search"]["search_count"] = self.search_count
        elif self.search_type == "google":
            config["search"]["source"] = "google"
            config["search"]["serpapi_key"] = self.serpapi_key or os.environ.get(
                "SERPAPI_KEY"
            )
            config["search"]["search_lang"] = self.search_lang
            config["search"]["search_count"] = self.search_count
        else:
            config["search"]["source"] = "aliyun"
            config["search"]["endpoint"] = self.aliyun_endpoint
            config["search"]["access_key_id"] = self.aliyun_access_key_id
            config["search"]["access_key_secret"] = self.aliyun_access_key_secret
            config["search"]["search_count"] = self.search_count

        config["guardrail"]["region"] = self.guardrail_region
        config["guardrail"]["endpoint"] = self.guardrail_endpoint
        config["guardrail"]["access_key_id"] = self.guardrail_ak
        config["guardrail"]["access_key_secret"] = self.guardrail_sk

        config["intent"]["descriptions"] = json.loads(self.intent_description)

        config["agent"]["system_prompt"] = self.agent_system_prompt
        config["agent"]["python_scripts"] = self.agent_python_scripts
        config["agent"]["function_definition"] = self.agent_function_definition
        config["agent"]["api_definition"] = self.agent_api_definition

        config["llms"] = self.llms

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
        settings["chat_model_id"] = {
            "choices": [
                llm.model_id if llm.model_id else llm.model for llm in self.llms
            ],
            "value": self.chat_model_id,
        }
        settings["query_rewrite_model_id"] = {
            "choices": [
                llm.model_id if llm.model_id else llm.model for llm in self.llms
            ],
            "value": self.query_rewrite_model_id,
        }
        settings["data_analysis_model_id"] = {
            "choices": [
                llm.model_id if llm.model_id else llm.model for llm in self.llms
            ],
            "value": self.data_analysis_model_id,
        }
        settings["rewrite_base_prompt"] = {"value": self.rewrite_base_prompt}
        settings["rewrite_agent_prompt"] = {"value": self.rewrite_agent_prompt}
        settings["rewrite_db_prompt"] = {"value": self.rewrite_db_prompt}
        settings["rewrite_knowledgebase_prompt"] = {
            "value": self.rewrite_knowledgebase_prompt
        }
        settings["rewrite_llm_prompt"] = {"value": self.rewrite_llm_prompt}
        settings["rewrite_search_prompt"] = {"value": self.rewrite_search_prompt}

        settings["use_oss"] = {"value": self.use_oss}
        settings["use_oss_col"] = {"visible": self.use_oss}

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

        settings["chunk_size"] = {"value": self.chunk_size}
        settings["chunk_overlap"] = {"value": self.chunk_overlap}
        settings["enable_multimodal"] = {"value": self.enable_multimodal}
        settings["enable_mandatory_ocr"] = {"value": self.enable_mandatory_ocr}
        settings["number_workers"] = {"value": self.number_workers}

        # retrieval and rerank
        settings["retrieval_mode"] = {"value": self.retrieval_mode}
        settings["reranker_type"] = {"value": self.reranker_type}
        settings["similarity_top_k"] = {"value": self.similarity_top_k}
        settings["image_similarity_top_k"] = {"value": self.image_similarity_top_k}
        settings["need_image"] = {"value": self.need_image}
        settings["reranker_model"] = {"value": self.reranker_model}
        settings["vector_weight"] = {
            "value": self.vector_weight,
            "visible": self.retrieval_mode == "混合检索",
        }
        settings["keyword_weight"] = {
            "value": self.keyword_weight,
            "visible": self.retrieval_mode == "混合检索",
        }
        settings["query_type"] = {
            "value": self.query_type,
        }
        settings["similarity_threshold"] = {"value": self.similarity_threshold}
        settings["reranker_similarity_threshold"] = {
            "value": self.reranker_similarity_threshold
        }
        settings["reranker_similarity_top_k"] = {
            "value": self.reranker_similarity_top_k
        }
        settings["model_reranker_col"] = {"visible": self.reranker_type == "基于模型的重排序"}

        settings["enable_query_transform"] = {
            "value": self.enable_query_transform,
        }
        settings["system_role_template"] = {
            "value": self.system_role_template,
        }
        settings["custom_prompt_template"] = {"value": self.custom_prompt_template}
        # settings["multimodal_qa_template"] = {"value": self.multimodal_qa_template}
        # settings["citation_multimodal_qa_template"] = {
        #     "value": self.citation_multimodal_qa_template
        # }

        # search
        settings["search_type"] = {"value": self.search_type}
        if self.search_type == "bing":
            settings["search_api_key"] = {"value": self.search_api_key, "visible": True}
            settings["search_lang"] = {"value": self.search_lang, "visible": True}
            settings["search_count"] = {"value": self.search_count, "visible": True}
            settings["serpapi_key"] = {"value": self.serpapi_key, "visible": False}
            settings["aliyun_endpoint"] = {
                "value": self.aliyun_endpoint,
                "visible": False,
            }
            settings["aliyun_access_key_id"] = {
                "value": self.aliyun_access_key_id,
                "visible": False,
            }
            settings["aliyun_access_key_secret"] = {
                "value": self.aliyun_access_key_secret,
                "visible": False,
            }
        elif self.search_type == "google":
            settings["search_api_key"] = {
                "value": self.search_api_key,
                "visible": False,
            }
            settings["search_lang"] = {"value": self.search_lang, "visible": True}
            settings["search_count"] = {"value": self.search_count, "visible": True}
            settings["serpapi_key"] = {"value": self.serpapi_key, "visible": True}
            settings["aliyun_endpoint"] = {
                "value": self.aliyun_endpoint,
                "visible": False,
            }
            settings["aliyun_access_key_id"] = {
                "value": self.aliyun_access_key_id,
                "visible": False,
            }
            settings["aliyun_access_key_secret"] = {
                "value": self.aliyun_access_key_secret,
                "visible": False,
            }
        # aliyun
        else:
            settings["search_api_key"] = {
                "value": self.search_api_key,
                "visible": False,
            }
            settings["search_lang"] = {"value": self.search_lang, "visible": False}
            settings["search_count"] = {"value": self.search_count, "visible": True}
            settings["serpapi_key"] = {"value": self.serpapi_key, "visible": False}
            settings["aliyun_endpoint"] = {
                "value": self.aliyun_endpoint,
                "visible": True,
            }
            settings["aliyun_access_key_id"] = {
                "value": self.aliyun_access_key_id,
                "visible": True,
            }
            settings["aliyun_access_key_secret"] = {
                "value": self.aliyun_access_key_secret,
                "visible": True,
            }

        # data_analysis
        settings["analysis_type"] = {"value": self.analysis_type}
        settings["analysis_file_path"] = {"value": self.analysis_file_path}
        settings["db_dialect"] = {"value": self.db_dialect}
        settings["db_username"] = {"value": self.db_username}
        settings["db_password"] = {"value": self.db_password}
        settings["db_host"] = {"value": self.db_host}
        settings["db_port"] = {"value": self.db_port}
        settings["database"] = {"value": self.database}
        settings["db_tables"] = {"value": self.db_tables}
        settings["db_descriptions"] = {"value": self.db_descriptions}
        settings["enable_enhanced_description"] = {
            "value": self.enable_enhanced_description
        }
        settings["enable_db_history"] = {"value": self.enable_db_history}
        settings["enable_db_embedding"] = {"value": self.enable_db_embedding}
        settings["max_col_num"] = {"value": self.max_col_num}
        settings["max_val_num"] = {"value": self.max_val_num}
        settings["enable_query_preprocessor"] = {
            "value": self.enable_query_preprocessor
        }
        settings["enable_db_preretriever"] = {"value": self.enable_db_preretriever}
        settings["enable_db_selector"] = {"value": self.enable_db_selector}
        settings["db_nl2sql_prompt"] = {"value": self.db_nl2sql_prompt}
        settings["synthesizer_prompt"] = {"value": self.synthesizer_prompt}

        # settings["da_llm_base_url"] = {
        #     "value": self.da_llm_base_url,
        # }
        # settings["da_llm_api_key"] = {
        #     "value": self.da_llm_api_key,
        # }
        # settings["da_llm_model_name"] = {
        #     "value": self.da_llm_model_name,
        # }
        # settings["da_llm_max_tokens"] = {
        #     "value": self.da_llm_max_tokens,
        # }

        settings["agent_system_prompt"] = {"value": self.agent_system_prompt}
        settings["agent_python_scripts"] = {"value": self.agent_python_scripts}
        settings["agent_api_definition"] = {"value": self.agent_api_definition}
        settings["agent_function_definition"] = {
            "value": self.agent_function_definition
        }

        settings["default_web_search"] = {"value": self.default_web_search}

        settings["intent_description"] = {"value": self.intent_description}

        settings["enable_guardrail"] = {"value": self.enable_guardrail}
        settings["guardrail_region"] = {"value": self.guardrail_region}
        settings["guardrail_endpoint"] = {"value": self.guardrail_endpoint}
        settings["guardrail_ak"] = {"value": self.guardrail_ak}
        settings["guardrail_sk"] = {"value": self.guardrail_sk}
        model_choices = [
            llm.model_id if llm.model_id else llm.model for llm in self.llms
        ]
        settings["llm_model"] = {
            "choices": model_choices + ["NEW"],
            "value": "NEW"
            if not self.llms and len(self.llms) == 0
            else model_choices[0],
        }

        # print("view model settings:", settings)

        return settings
