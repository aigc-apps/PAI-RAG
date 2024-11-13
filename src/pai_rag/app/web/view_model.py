from pydantic import BaseModel
from typing import Any, Dict
from collections import defaultdict
from pai_rag.app.web.ui_constants import (
    LLM_MODEL_KEY_DICT,
    MLLM_MODEL_KEY_DICT,
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL,
)
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
from pai_rag.integrations.llms.pai.llm_config import DashScopeLlmConfig, PaiEasLlmConfig
from pai_rag.integrations.postprocessor.pai.pai_postprocessor import (
    SimilarityPostProcessorConfig,
)


def recursive_dict():
    return defaultdict(recursive_dict)


def _transform_to_dict(config):
    for k, v in config.items():
        if isinstance(v, defaultdict):
            config[k] = _transform_to_dict(v)
    return dict(config)


class ViewModel(BaseModel):
    # llm
    llm: str = "PaiEas"
    llm_eas_url: str = None
    llm_eas_token: str = None
    llm_eas_model_name: str = "default"
    llm_api_key: str = None
    llm_api_model_name: str = None
    llm_temperature: float = 0.1

    # mllm
    use_mllm: bool = False
    mllm: str = None
    mllm_eas_url: str = None
    mllm_eas_token: str = None
    mllm_eas_model_name: str = "default"
    mllm_api_key: str = None
    mllm_api_model_name: str = None

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
    enable_raptor: bool = False
    enable_table_summary: bool = False

    config_file: str = None

    vectordb_type: str = "faiss"

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
    database: str = None
    db_tables: str = None
    db_descriptions: str = None
    db_nl2sql_prompt: str = None
    synthesizer_prompt: str = None

    # postprocessor
    reranker_type: str = "no-reranker"  # no-reranker / model-based-reranker
    reranker_model: str = "bge-reranker-base"  # bge-reranker-base / bge-reranker-large
    keyword_weight: float = 0.3
    vector_weight: float = 0.7
    similarity_threshold: float = 0.5
    reranker_similarity_threshold: float = 0

    query_engine_type: str = None

    synthesizer_type: str = None

    text_qa_template: str = DEFAULT_TEXT_QA_PROMPT_TMPL
    multimodal_qa_template: str = DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL

    # agent
    agent_api_definition: str = None  # API tool definition
    agent_function_definition: str = None  # Function tool definition
    agent_python_scripts: str = None  # Function scripts
    agent_system_prompt: str = None  # Agent system prompt

    # intent
    intent_description: str = None

    def update(self, update_paras: Dict[str, Any]):
        attr_set = set(dir(self))
        for key, value in update_paras.items():
            if key in attr_set:
                setattr(self, key, value)

    @staticmethod
    def from_app_config(config: RagConfig):
        view_model = ViewModel()

        # llm
        view_model.llm = config.llm.source.value
        if isinstance(config.llm, PaiEasLlmConfig):
            view_model.llm_eas_model_name = config.llm.model
            view_model.llm_eas_url = config.llm.endpoint
            view_model.llm_eas_token = config.llm.token
        elif isinstance(config.llm, DashScopeLlmConfig):
            view_model.llm_api_key = config.llm.api_key
            view_model.llm_api_model_name = config.llm.model

        view_model.use_mllm = config.synthesizer.use_multimodal_llm

        view_model.mllm = config.multimodal_llm.source.value
        if isinstance(config.multimodal_llm, PaiEasLlmConfig):
            view_model.mllm_eas_url = config.multimodal_llm.endpoint
            view_model.mllm_eas_model_name = config.multimodal_llm.model
            view_model.mllm_eas_token = config.multimodal_llm.token
        else:
            view_model.mllm_api_model_name = config.multimodal_llm.model
            view_model.mllm_api_key = config.multimodal_llm.api_key

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

        view_model.enable_table_summary = config.data_reader.enable_table_summary

        view_model.similarity_top_k = config.retriever.similarity_top_k
        view_model.image_similarity_top_k = config.retriever.image_similarity_top_k
        view_model.need_image = config.retriever.search_image
        view_model.vector_weight = config.retriever.hybrid_fusion_weights[0]
        view_model.keyword_weight = config.retriever.hybrid_fusion_weights[1]
        if config.retriever.vector_store_query_mode == VectorStoreQueryMode.DEFAULT:
            view_model.retrieval_mode = "Embedding Only"
        elif config.retriever.vector_store_query_mode == VectorStoreQueryMode.HYBRID:
            view_model.retrieval_mode = "Hybrid"
        else:
            view_model.retrieval_mode = "Keyword Only"

        view_model.reranker_type = config.postprocessor.reranker_type.value
        if isinstance(config.postprocessor, SimilarityPostProcessorConfig):
            view_model.similarity_threshold = config.postprocessor.similarity_threshold
        else:
            view_model.reranker_model = config.postprocessor.reranker_model
            view_model.reranker_similarity_threshold = (
                config.postprocessor.similarity_threshold
            )

        view_model.text_qa_template = config.synthesizer.text_qa_template
        view_model.multimodal_qa_template = config.synthesizer.multimodal_qa_template

        view_model.search_api_key = config.search.search_api_key or os.environ.get(
            "BING_SEARCH_KEY"
        )
        view_model.search_lang = config.search.search_lang
        view_model.search_count = config.search.search_count

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
            view_model.db_descriptions = json.dumps(
                config.data_analysis.descriptions, ensure_ascii=False
            )
        view_model.db_nl2sql_prompt = config.data_analysis.nl2sql_prompt
        view_model.synthesizer_prompt = config.data_analysis.synthesizer_prompt

        view_model.agent_api_definition = config.agent.api_definition
        view_model.agent_function_definition = config.agent.function_definition
        view_model.agent_python_scripts = config.agent.python_scripts
        view_model.agent_system_prompt = config.agent.system_prompt

        view_model.intent_description = json.dumps(
            config.intent.descriptions, ensure_ascii=False, sort_keys=True, indent=4
        )

        return view_model

    def to_app_config(self):
        config = recursive_dict()

        config["llm"]["source"] = self.llm
        config["llm"]["endpoint"] = self.llm_eas_url
        config["llm"]["token"] = self.llm_eas_token
        config["llm"]["api_key"] = self.llm_api_key
        config["llm"]["temperature"] = self.llm_temperature
        if self.llm.lower() == "paieas":
            config["llm"]["model"] = self.llm_eas_model_name
        else:
            config["llm"]["model"] = self.llm_api_model_name

        config["multimodal_llm"]["source"] = self.mllm
        config["multimodal_llm"]["endpoint"] = self.mllm_eas_url
        config["multimodal_llm"]["token"] = self.mllm_eas_token
        config["multimodal_llm"]["api_key"] = self.mllm_api_key
        if self.mllm.lower() == "paieas":
            config["multimodal_llm"]["model"] = self.mllm_eas_model_name
        else:
            config["multimodal_llm"]["model"] = self.mllm_api_model_name

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

        config["data_reader"]["enable_table_summary"] = self.enable_table_summary

        config["retriever"]["similarity_top_k"] = self.similarity_top_k
        config["retriever"]["image_similarity_top_k"] = self.image_similarity_top_k
        config["retriever"]["vector_weight"] = self.vector_weight
        config["retriever"]["keyword_weight"] = self.keyword_weight

        config["retriever"]["image_similarity_top_k"] = self.image_similarity_top_k

        config["retriever"]["search_image"] = self.need_image
        if self.retrieval_mode == "Hybrid":
            config["retriever"]["vector_store_query_mode"] = VectorStoreQueryMode.HYBRID
        elif self.retrieval_mode == "Embedding Only":
            config["retriever"][
                "vector_store_query_mode"
            ] = VectorStoreQueryMode.DEFAULT
        elif self.retrieval_mode == "Keyword Only":
            config["retriever"][
                "vector_store_query_mode"
            ] = VectorStoreQueryMode.TEXT_SEARCH

        if self.analysis_type == "nl2pandas":
            config["data_analysis"]["type"] = "pandas"
            config["data_analysis"]["analysis_file_path"] = self.analysis_file_path
        elif self.analysis_type == "nl2sql":
            config["data_analysis"]["type"] = "mysql"
            config["data_analysis"]["user"] = self.db_username
            config["data_analysis"]["password"] = self.db_password
            config["data_analysis"]["host"] = self.db_host
            config["data_analysis"]["port"] = self.db_port
            config["data_analysis"]["database"] = self.database
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

        config["postprocessor"]["reranker_type"] = self.reranker_type
        config["postprocessor"]["reranker_model"] = self.reranker_model
        if self.reranker_type == "no-reranker":
            config["postprocessor"]["similarity_threshold"] = self.similarity_threshold
        else:
            config["postprocessor"][
                "similarity_threshold"
            ] = self.reranker_similarity_threshold
        config["postprocessor"]["top_n"] = self.similarity_top_k

        config["synthesizer"]["use_multimodal_llm"] = self.use_mllm
        config["synthesizer"]["text_qa_template"] = self.text_qa_template
        config["synthesizer"]["multimodal_qa_template"] = self.multimodal_qa_template

        config["search"]["search_api_key"] = self.search_api_key or os.environ.get(
            "BING_SEARCH_KEY"
        )
        config["search"]["search_lang"] = self.search_lang
        config["search"]["search_count"] = self.search_count

        config["intent"]["descriptions"] = json.loads(self.intent_description)

        config["agent"]["system_prompt"] = self.agent_system_prompt
        config["agent"]["python_scripts"] = self.agent_python_scripts
        config["agent"]["function_definition"] = self.agent_function_definition
        config["agent"]["api_definition"] = self.agent_api_definition

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
        settings["llm"] = {"value": self.llm}
        settings["llm_eas_url"] = {
            "value": self.llm_eas_url,
            "visible": self.llm.lower() == "paieas",
        }
        settings["llm_eas_token"] = {
            "value": self.llm_eas_token,
            "visible": self.llm.lower() == "paieas",
        }
        settings["llm_api_key"] = {
            "value": self.llm_api_key,
            "visible": self.llm.lower() != "paieas",
        }
        if self.llm.lower() == "paieas" and not self.llm_eas_model_name:
            self.llm_eas_model_name = "default"

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
        settings["use_mllm_col"] = {"visible": self.use_mllm}

        settings["mllm"] = {"value": self.mllm}
        settings["mllm_eas_url"] = {"value": self.mllm_eas_url}
        settings["mllm_eas_token"] = {"value": self.mllm_eas_token}
        settings["mllm_eas_model_name"] = {"value": self.mllm_eas_model_name}
        settings["mllm_api_model_name"] = {
            "value": self.mllm_api_model_name,
            "choices": MLLM_MODEL_KEY_DICT.get(self.mllm, []),
            "visible": self.mllm.lower() != "paieas",
        }
        settings["mllm_api_key"] = {
            "value": self.mllm_api_key,
            "visible": self.mllm.lower() != "paieas",
        }
        settings["m_eas_col"] = {"visible": self.mllm == "paieas"}
        settings["api_mllm_col"] = {"visible": self.mllm == "dashscope"}

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
        settings["vector_weight"] = {
            "value": self.vector_weight,
            "visible": self.retrieval_mode == "Hybrid",
        }
        settings["keyword_weight"] = {
            "value": self.keyword_weight,
            "visible": self.retrieval_mode == "Hybrid",
        }
        settings["similarity_threshold"] = {"value": self.similarity_threshold}
        settings["reranker_similarity_threshold"] = {
            "value": self.reranker_similarity_threshold
        }

        settings["text_qa_template"] = {"value": self.text_qa_template}
        settings["multimodal_qa_template"] = {"value": self.multimodal_qa_template}

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
        settings["database"] = {"value": self.database}
        settings["db_tables"] = {"value": self.db_tables}
        settings["db_descriptions"] = {"value": self.db_descriptions}
        settings["db_nl2sql_prompt"] = {"value": self.db_nl2sql_prompt}
        settings["synthesizer_prompt"] = {"value": self.synthesizer_prompt}

        settings["agent_system_prompt"] = {"value": self.agent_system_prompt}
        settings["agent_python_scripts"] = {"value": self.agent_python_scripts}
        settings["agent_api_definition"] = {"value": self.agent_api_definition}
        settings["agent_function_definition"] = {
            "value": self.agent_function_definition
        }

        settings["intent_description"] = {"value": self.intent_description}

        return settings
