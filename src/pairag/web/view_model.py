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
from pairag.core.rag_config import RagConfig
from pairag.integrations.data_analysis.data_analysis_config import (
    MysqlAnalysisConfig,
    PandasAnalysisConfig,
    SqliteAnalysisConfig,
)
from pairag.integrations.llms.pai.llm_config import (
    OpenAICompatibleLlmConfig,
)
from pairag.integrations.search.search_config import (
    DEFAULT_ALIYUN_SEARCH_ENDPOINT,
    DEFAULT_SEARCH_COUNT,
    BingSearchConfig,
    AliyunSearchConfig,
    GoogleSearchConfig,
)
from pairag.integrations.postprocessor.pai.pai_postprocessor import PostProcessorType

# deprecated
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

# new version of query_type
QUERY_TYPES_MAP = {
    "大模型": "chat_llm",
    "联网搜索": "search_web",
    "查询知识库": "chat_knowledgebase",
    "查询数据库": "chat_db",
    "新闻工具": "chat_news",
}

INVERTED_QUERY_TYPES_MAP = {
    "chat_llm": "大模型",
    "search_web": "联网搜索",
    "chat_knowledgebase": "查询知识库",
    "chat_db": "查询数据库",
    "chat_news": "新闻工具",
}

RETRIEVAL_MODE_MAP = {
    "向量检索": VectorStoreQueryMode.DEFAULT,
    "关键字检索": VectorStoreQueryMode.TEXT_SEARCH,
    "混合检索": VectorStoreQueryMode.HYBRID,
}

INVERTED_RETRIEVAL_MODE_MAP = {
    VectorStoreQueryMode.DEFAULT: "向量检索",
    VectorStoreQueryMode.TEXT_SEARCH: "关键字检索",
    VectorStoreQueryMode.HYBRID: "混合检索",
}

RERANKER_TYPE_MAP = {
    "无重排序": PostProcessorType.no_reranker,
    "基于模型的重排序": PostProcessorType.reranker_model,
}

INVERTED_RERANKER_TYPE_MAP = {
    PostProcessorType.no_reranker: "无重排序",
    PostProcessorType.reranker_model: "基于模型的重排序",
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

    # websearch
    search_type: str = "bing"
    search_api_key: str = None
    search_count: int = DEFAULT_SEARCH_COUNT
    search_lang: str = "zh-CN"
    search_role_template: str = None
    search_qa_prompt_template: str = ""

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

    query_type: str = "对话 (知识库)"  # deprecated
    query_types: List = ["大模型"]

    enable_query_transform: bool = True
    rewrite_base_prompt: str = None
    rewrite_only_prompt: str = None
    rewrite_llm_prompt: str = None
    rewrite_knowledgebase_prompt: str = None
    rewrite_agent_prompt: str = None
    rewrite_search_prompt: str = None
    rewrite_db_prompt: str = None
    rewrite_news_prompt: str = None

    synthesizer_type: str = None
    system_role_template: str = None
    custom_prompt_template: str = None
    # multimodal_qa_template: str = None
    # citation_multimodal_qa_template: str = None

    # guardrail
    guardrail_ak: str = None
    guardrail_sk: str = None
    guardrail_endpoint: str = None
    guardrail_region: str = None
    enable_guardrail: bool = False
    guardrail_advice: str = None

    # llms
    llms: List[OpenAICompatibleLlmConfig] = None

    # news_extension
    news_extension_model_id: str = "default"
    bailian_workspaceid: str = None
    bailian_ak: str = None
    bailian_sk: str = None
    top_news_count: int = 10
    list_news_pmt: str = None
    chat_news_model_id: str = "qwen-max-latest"
    chat_news_pmt: str = None
    domain_list: str = None
    news_role: str = None
    trace_app_name: str = None
    telemetry_endpoint: str = None
    telemetry_token: str = None
    telemetry_enabled: bool = None
    trace_args_mapping_str: str = None

    def update(self, update_paras: Dict[str, Any]):
        attr_set = set(dir(self))
        for key, value in update_paras.items():
            if key in attr_set:
                setattr(self, key, value)

    @staticmethod
    def from_app_config(config: RagConfig):
        view_model = ViewModel()

        view_model.llms = config.llms

        view_model.chat_model_id = config.chat.model_id or "default"

        view_model.query_type = INVERTED_QUERY_TYPE_MAP.get(
            config.system.query_type, "对话 (知识库)"
        )  # deprecated
        view_model.query_types = [
            INVERTED_QUERY_TYPES_MAP.get(qt, "大模型") for qt in config.system.query_types
        ]

        view_model.use_oss = (
            config.oss_store.bucket is not None and config.oss_store.bucket != ""
        )
        view_model.oss_ak = config.oss_store.ak
        view_model.oss_sk = config.oss_store.sk
        view_model.oss_endpoint = config.oss_store.endpoint
        view_model.oss_bucket = config.oss_store.bucket

        view_model.enable_query_transform = config.query_rewrite.enabled
        view_model.rewrite_base_prompt = config.query_rewrite.base_prompt_template_str
        view_model.rewrite_only_prompt = config.query_rewrite.rewrite_only_prompt_str
        view_model.rewrite_llm_prompt = config.query_rewrite.llm_tool_prompt_str
        view_model.rewrite_knowledgebase_prompt = (
            config.query_rewrite.knowledge_tool_prompt_str
        )

        view_model.rewrite_search_prompt = (
            config.query_rewrite.websearch_tool_prompt_str
        )
        view_model.rewrite_news_prompt = config.query_rewrite.news_tool_prompt_str

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
        view_model.search_qa_prompt_template = config.search.search_qa_prompt_template
        view_model.search_role_template = config.search.search_role_template
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

        if config.guardrail.is_enabled():
            view_model.enable_guardrail = True
            view_model.guardrail_ak = config.guardrail.access_key_id
            view_model.guardrail_sk = config.guardrail.access_key_secret
            view_model.guardrail_endpoint = config.guardrail.endpoint
            view_model.guardrail_region = config.guardrail.region
            view_model.guardrail_advice = config.guardrail.custom_advice

        # news_extension
        view_model.news_extension_model_id = config.news_extension.model_id or "default"
        view_model.chat_news_model_id = config.news_extension.chat_news_model_id
        view_model.bailian_workspaceid = config.news_extension.workspace_id
        view_model.bailian_ak = config.news_extension.access_key_id
        view_model.bailian_sk = config.news_extension.access_key_secret
        view_model.top_news_count = config.news_extension.top_news_count
        view_model.list_news_pmt = config.news_extension.list_topics_prompt_str
        # view_model.chat_news_answer_len = config.news_extension.chat_news_answer_len
        view_model.chat_news_pmt = config.news_extension.chat_news_prompt_str
        view_model.domain_list = ",".join(config.news_extension.domain_list)
        view_model.news_role = config.news_extension.news_role

        view_model.trace_app_name = config.trace.service_name
        view_model.telemetry_endpoint = config.trace.endpoint
        view_model.telemetry_token = config.trace.token
        view_model.telemetry_enabled = config.trace.enabled
        view_model.trace_args_mapping_str = json.dumps(config.trace.user_args)

        return view_model

    def to_app_config(self):
        config = recursive_dict()

        config["system"]["query_type"] = QUERY_TYPE_MAP.get(
            self.query_type, "rag"
        )  # deprecated
        config["system"]["query_types"] = [
            QUERY_TYPES_MAP.get(qt, "chat_llm") for qt in self.query_types
        ]

        config["chat"]["model_id"] = self.chat_model_id
        config["query_rewrite"]["model_id"] = self.query_rewrite_model_id
        config["data_analysis"]["model_id"] = self.data_analysis_model_id

        if os.getenv("OSS_ACCESS_KEY_ID") is None and self.oss_ak:
            os.environ["OSS_ACCESS_KEY_ID"] = self.oss_ak
        if os.getenv("OSS_ACCESS_KEY_SECRET") is None and self.oss_sk:
            os.environ["OSS_ACCESS_KEY_SECRET"] = self.oss_sk
        config["oss_store"]["ak"] = self.oss_ak
        config["oss_store"]["sk"] = self.oss_sk
        config["oss_store"]["endpoint"] = self.oss_endpoint
        config["oss_store"]["bucket"] = self.oss_bucket

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

        config["synthesizer"]["custom_prompt_template"] = self.custom_prompt_template
        config["synthesizer"]["system_role_template"] = self.system_role_template

        config["query_rewrite"]["base_prompt_template_str"] = self.rewrite_base_prompt
        config["query_rewrite"]["rewrite_only_prompt_str"] = self.rewrite_only_prompt
        config["query_rewrite"]["llm_tool_prompt_str"] = self.rewrite_llm_prompt
        config["query_rewrite"][
            "knowledge_tool_prompt_str"
        ] = self.rewrite_knowledgebase_prompt
        config["query_rewrite"][
            "websearch_tool_prompt_str"
        ] = self.rewrite_search_prompt
        config["query_rewrite"]["news_tool_prompt_str"] = self.rewrite_news_prompt
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
        config["search"]["search_qa_prompt_template"] = self.search_qa_prompt_template
        config["search"]["search_role_template"] = self.search_role_template

        config["guardrail"]["region"] = self.guardrail_region
        config["guardrail"]["endpoint"] = self.guardrail_endpoint
        config["guardrail"]["access_key_id"] = self.guardrail_ak
        config["guardrail"]["access_key_secret"] = self.guardrail_sk
        config["guardrail"]["custom_advice"] = self.guardrail_advice

        config["llms"] = [llm.model_dump(mode="json") for llm in self.llms]

        # news_extension
        config["news_extension"]["workspace_id"] = self.bailian_workspaceid
        config["news_extension"]["access_key_id"] = self.bailian_ak
        config["news_extension"]["access_key_secret"] = self.bailian_sk
        config["news_extension"]["model_id"] = self.news_extension_model_id
        config["news_extension"]["top_news_count"] = self.top_news_count
        config["news_extension"]["list_topics_prompt_str"] = self.list_news_pmt
        config["news_extension"]["chat_news_model_id"] = self.chat_news_model_id
        config["news_extension"]["chat_news_prompt_str"] = self.chat_news_pmt
        config["news_extension"]["domain_list"] = self.domain_list.split(",")
        config["news_extension"]["news_role"] = self.news_role

        config["trace"]["service_name"] = self.trace_app_name
        config["trace"]["endpoint"] = self.telemetry_endpoint
        config["trace"]["token"] = self.telemetry_token
        config["trace"]["enabled"] = self.telemetry_enabled
        config["trace"]["user_args"] = json.loads(self.trace_args_mapping_str)

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
        settings["rewrite_only_prompt"] = {"value": self.rewrite_only_prompt}

        settings["rewrite_agent_prompt"] = {"value": self.rewrite_agent_prompt}
        settings["rewrite_db_prompt"] = {"value": self.rewrite_db_prompt}
        settings["rewrite_knowledgebase_prompt"] = {
            "value": self.rewrite_knowledgebase_prompt
        }
        settings["rewrite_llm_prompt"] = {"value": self.rewrite_llm_prompt}
        settings["rewrite_search_prompt"] = {"value": self.rewrite_search_prompt}
        settings["rewrite_news_prompt"] = {"value": self.rewrite_news_prompt}

        settings["use_oss"] = {"value": self.use_oss}
        settings["use_oss_col"] = {"visible": self.use_oss}

        settings["oss_ak"] = {
            "value": self.oss_ak if self.oss_ak else self.oss_ak,
            "type": "text",
        }
        settings["oss_sk"] = {
            "value": self.oss_sk if self.oss_sk else self.oss_sk,
            "type": "password",
        }
        settings["oss_endpoint"] = {"value": self.oss_endpoint}
        settings["oss_bucket"] = {"value": self.oss_bucket}
        settings["query_type"] = {
            "value": self.query_type,
        }  # deprecated
        settings["query_types"] = {
            "value": self.query_types,
        }

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
        settings["search_qa_prompt_template"] = {
            "value": self.search_qa_prompt_template,
            "visible": True,
        }
        settings["search_role_template"] = {
            "value": self.search_role_template,
            "visible": True,
        }

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

        settings["enable_guardrail"] = {"value": self.enable_guardrail}
        settings["guardrail_col"] = {"visible": self.enable_guardrail}
        settings["guardrail_advice"] = {"value": self.guardrail_advice}
        settings["guardrail_region"] = {"value": self.guardrail_region}
        settings["guardrail_endpoint"] = {"value": self.guardrail_endpoint}
        settings["guardrail_ak"] = {"value": self.guardrail_ak}
        settings["guardrail_sk"] = {"value": self.guardrail_sk}
        model_choices = [
            llm.model_id if llm.model_id else llm.model for llm in self.llms
        ]
        settings["llm_model"] = {
            "choices": model_choices + ["NEW"],
            "value": "NEW" if not self.llms else model_choices[0],
        }

        # news_extension
        settings["news_extension_model_id"] = {
            "choices": [
                llm.model_id if llm.model_id else llm.model for llm in self.llms
            ],
            "value": self.news_extension_model_id,
        }
        settings["bailian_workspaceid"] = {"value": self.bailian_workspaceid}
        settings["bailian_ak"] = {"value": self.bailian_ak}
        settings["bailian_sk"] = {"value": self.bailian_sk}
        settings["top_news_count"] = {"value": self.top_news_count}
        settings["list_news_pmt"] = {"value": self.list_news_pmt}
        settings["chat_news_model_id"] = {"value": self.chat_news_model_id}
        settings["chat_news_pmt"] = {"value": self.chat_news_pmt}
        settings["domain_list"] = {"value": self.domain_list}
        settings["news_role"] = {"value": self.news_role}
        # print("view model settings:", settings)

        settings["trace_app_name"] = {"value": self.trace_app_name}
        settings["telemetry_endpoint"] = {"value": self.telemetry_endpoint}
        settings["telemetry_token"] = {"value": self.telemetry_token}
        settings["telemetry_enabled"] = {"value": self.telemetry_enabled}
        settings["trace_args_mapping_str"] = {
            "value": self.trace_args_mapping_str or "{}",
        }

        settings["llm_model_name"] = {
            "value": self.llms[0].model if self.llms else "",
        }
        settings["llm_base_url"] = {
            "value": self.llms[0].base_url if self.llms else "",
        }
        settings["llm_api_key"] = {
            "value": self.llms[0].api_key if self.llms else "",
        }
        settings["llm_model_id"] = {
            "value": self.llms[0].model_id if self.llms else "",
        }
        settings["llm_model_context_window"] = {
            "value": self.llms[0].context_window if self.llms else "",
        }
        settings["llm_model_max_tokens"] = {
            "value": self.llms[0].max_tokens if self.llms else "",
        }
        settings["llm_vision_support"] = {
            "value": self.llms[0].vision_support if self.llms else "",
        }
        settings["llm_reasoning_support"] = {
            "value": self.llms[0].is_reasoning_model if self.llms else "",
        }
        settings["llm_model_temperature"] = {
            "value": self.llms[0].temperature if self.llms else 0.1,
        }
        settings["llm_extra_kwargs_str"] = {
            "value": self.llms[0].extra_body_str if self.llms else "",
        }

        return settings
