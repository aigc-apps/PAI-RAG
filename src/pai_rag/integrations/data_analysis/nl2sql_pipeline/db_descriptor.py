import logging
import json
from typing import Any, Callable, Dict, List, Optional, Union, cast

from pydantic import BaseModel, Field
from sqlalchemy import Table

from llama_index.core.llms.llm import LLM
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.objects.table_node_mapping import SQLTableSchema
from llama_index.core.schema import QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core import Settings
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core import PromptTemplate, BasePromptTemplate
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_SUMMARY_PROMPT = PromptTemplate(
    "下面是用户数据库{db_name}中各个数据表结构信息以及抽样数据样例。\n"
    "表结构信息及数据样例: {db_schema_info}, \n\n"
    "请学习理解该数据的结构和内容，按要求输出解析结果: \n"
    "分析每个数据表中各列数据的含义和作用，并对专业术语进行简单明了的解释。\n"
    "如果是时间类型请给出时间格式，类似:yyyy-MM-dd HH:MM:ss或者yyyy-MM等。\n"
    "请不要修改或者翻译列名，确保和给出数据列名一致。\n"
    "针对数据从不同维度提供一些有用的分析思路给用户。\n\n"
    "请一步一步思考，以中文回答。\n"
    "回答: "
)

DEFAULT_FILE_PATH = "TODO"


class DBDescriptor(PromptMixin):
    """
    收集数据库结构信息(表名、列名、采样数据等): DDL + data sample
    基于结构信息利用LLM进行信息总结: db_description/analysis
    收集历史查询记录: sql_history
    """

    def __init__(
        self,
        sql_database: SQLDatabase,
        tables: List[str] = None,
        dialect: str = "mysql",
        db_name: str = None,
        context_query_kwargs: Optional[dict] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
        context_str_prefix: Optional[str] = None,
        llm: Optional[LLM] = None,
        service_context: Optional[ServiceContext] = None,
        db_summary_prompt: Optional[BasePromptTemplate] = None,
    ):
        self._sql_database = sql_database
        self._tables = tables
        self._dialect = dialect
        self._dbname = db_name
        self._get_tables = self._load_get_tables_fn(
            sql_database, tables, context_query_kwargs, table_retriever
        )
        self._context_str_prefix = context_str_prefix
        self._llm = llm or Settings.llm
        self._db_summary_prompt = db_summary_prompt or DEFAULT_DB_SUMMARY_PROMPT

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "db_summary_prompt": self._db_summary_prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "db_summary_prompt" in prompts:
            self._db_summary_prompt = prompts["db_summary_prompt"]

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def collect_schema(self, query_bundle: QueryBundle) -> str:
        # Get tables schema + optional context + data sample as a single string.

        table_schema_objs = self._get_tables(
            query_bundle.query_str
        )  # get a list of SQLTableSchema, e.g. [SQLTableSchema(table_name='has_pet', context_str=None),]
        context_strs = []
        if self._context_str_prefix is not None:
            context_strs = [self._context_str_prefix]

        for table_schema_obj in table_schema_objs:
            table_info = self._sql_database.get_single_table_info(
                table_schema_obj.table_name
            )  # get ddl info
            data_sample = self._get_data_sample(
                table_schema_obj.table_name
            )  # get data sample
            table_info_with_sample = table_info + "\ndata_sample: " + data_sample

            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info_with_sample += table_opt_context

            context_strs.append(table_info_with_sample)

        return "\n\n".join(context_strs)

    def _load_get_tables_fn(
        self,
        sql_database: SQLDatabase,
        tables: Optional[Union[List[str], List[Table]]] = None,
        context_query_kwargs: Optional[dict] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
    ) -> Callable[[str], List[SQLTableSchema]]:
        # Load get_tables function
        context_query_kwargs = context_query_kwargs or {}
        if table_retriever is not None:
            return lambda query_str: cast(Any, table_retriever).retrieve(query_str)
        else:
            if tables is not None:
                table_names: List[str] = [
                    t.name if isinstance(t, Table) else t for t in tables
                ]
            else:
                table_names = list(sql_database.get_usable_table_names())
            context_strs = [context_query_kwargs.get(t, None) for t in table_names]
            table_schemas = [
                SQLTableSchema(table_name=t, context_str=c)
                for t, c in zip(table_names, context_strs)
            ]
            return lambda _: table_schemas

    def _get_data_sample(self, table: str, sample_n: int = 3) -> str:
        # 对每个table随机采样
        if self._dialect == "mysql":
            sql_str = f"SELECT * FROM {table} ORDER BY RAND() LIMIT {sample_n};"
        if self._dialect in ("sqlite", "postgresql"):
            sql_str = f"Select * FROM {table} ORDER BY RANDOM() LIMIT {sample_n};"
        table_sample, _ = self._sql_database.run_sql(sql_str)

        return table_sample

    def summarize_schema(self, file_path: Optional[str] = None):
        # 利用LLM总结schema info，同步
        db_schema_info = self.collect_schema(QueryBundle(""))
        sllm = self._llm.as_structured_llm(output_cls=AnalysisOutput)
        output_summary = sllm.predict(
            prompt=self._db_summary_prompt,
            db_name=self._dbname,
            db_schema_info=db_schema_info,
        )

        parsed_json = json.loads(output_summary)
        structured_output_summary = json.dumps(
            parsed_json, indent=4, ensure_ascii=False
        )

        # 保存为txt文件
        if file_path is None:
            file_path = DEFAULT_FILE_PATH
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(structured_output_summary)

        return structured_output_summary

    async def asummarize_schema(self, file_path: Optional[str] = None):
        # 利用LLM总结schema info，异步
        db_schema_info = self.collect_schema(QueryBundle(""))
        sllm = self._llm.as_structured_llm(output_cls=AnalysisOutput)
        output_summary = await sllm.apredict(
            prompt=self._db_summary_prompt,
            db_name=self._dbname,
            db_schema_info=db_schema_info,
        )

        parsed_json = json.loads(output_summary)
        structured_output_summary = json.dumps(
            parsed_json, indent=4, ensure_ascii=False
        )

        # 保存为txt文件
        if file_path is None:
            file_path = DEFAULT_FILE_PATH
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(structured_output_summary)

        return structured_output_summary

    def collect_history(self, log_path: str):
        # 收集历史查询记录
        pass

    def knowledge_index(
        self,
    ):
        """
        description 切分chunk，embedding，store
        history 是否处理？
        db_value 是否处理？
        """
        pass


# class TableColumnIntro(BaseModel):
#     """Data model for ColumnIntro"""
#     table_name: str
#     column_name: str
#     column_type: str = Field(description="字段的类型，如INTEGER, VARCHAR(20), REAL等")
#     column_description: str = Field(description="字段的描述，包括专业术语解释，时间格式解释等")
#     column_sample: list[str] = Field(description="从提供的data_sample中列举2个例子，取值尽可能不同")


# class AnalysisOutput(BaseModel):
#     """Data model for AnalysisOutput."""

#     data_analysis: str = Field(description="数据内容分析总结")
#     analysis_program: List[str] = Field(description="可能的分析方案,列举3个")
#     table_column_analysis: List[TableColumnIntro] = Field(description="仅限于从提供的数据表结构中介绍每张表每个字段,专业术语解释,以及2条数据样例, 尽量简单明了,禁止生成提供信息以外的内容")


class AnalysisOutput(BaseModel):
    """Data model for AnalysisOutput."""

    data_analysis: str = Field(description="数据内容分析总结")
    analysis_program: List[str] = Field(description="可能的分析方案,列举3个")
    table_column_analysis: List[str] = Field(
        description="基于提供的数据表结构信息和样本信息, 提供每个字段的补充描述和2条数据样本, 具体格式为: 表名|字段名|字段类型|字段描述|2条数据样本(以英文逗号分割), \n其中字段描述可以包括专业术语解释、时间格式解释等, 禁止生成提供信息以外的内容"
    )
