import logging
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy import Table

from llama_index.core.llms.llm import LLM
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.objects.table_node_mapping import SQLTableSchema
from llama_index.core.schema import QueryBundle
from llama_index.core.service_context import ServiceContext
from llama_index.core import Settings
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core import BasePromptTemplate
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)

from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_DB_SUMMARY_PROMPT,
)


logger = logging.getLogger(__name__)

DEFAULT_DESC_FILE_PATH = (
    "./localdata/data_analysis/nl2sql/db_structured_description.txt"
)
DEFAULT_HISTORY_FILE_PATH = "./localdata/data_analysis/nl2sql/db_query_history.txt"


class DBDescriptor(PromptMixin):
    """
    收集数据库结构信息(表名、列名、采样数据等): DDL + data sample + opt description
    基于结构信息利用LLM进行信息总结补充
    收集历史查询记录: sql_history
    """

    def __init__(
        self,
        sql_database: SQLDatabase,
        db_name: str = None,
        context_query_kwargs: Optional[dict] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
        context_str_prefix: Optional[str] = None,
        llm: Optional[LLM] = None,
        service_context: Optional[ServiceContext] = None,
        db_summary_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        self._sql_database = sql_database
        self._tables = list(sql_database._usable_tables)
        self._dialect = sql_database.dialect
        self._dbname = db_name
        self._get_tables = self._load_get_tables_fn(
            sql_database, self._tables, context_query_kwargs, table_retriever
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

    def get_structured_table_description(
        self, query_bundle: QueryBundle, file_path: Optional[str] = None
    ) -> Tuple[List, List]:
        """
        Get structured table schema with data samples + optional context description from webui
        """
        table_schema_objs = self._get_tables(
            query_bundle.query_str
        )  # get a list of SQLTableSchema, e.g. [SQLTableSchema(table_name='has_pet', context_str=None),]
        table_col_list = []
        table_foreign_key_list = []
        table_opt_context_list = []
        for table_schema_obj in table_schema_objs:
            table_name = table_schema_obj.table_name

            # collect optional context descriptions from webui
            if table_schema_obj.context_str:
                table_opt_context_list.append(
                    {"table": table_name, "description": table_schema_obj.context_str}
                )

            # collect data samples
            data_sample = self._get_data_sample(table_name)

            # collect and structure table schema with data samples
            table_pk = self._sql_database._inspector.get_pk_constraint(
                table_name, self._sql_database._schema
            )  # get primary key
            if len(table_pk["constrained_columns"]) > 0:
                table_pk_col = table_pk["constrained_columns"][0]
            else:
                table_pk_col = None

            for i, col in enumerate(
                self._sql_database._inspector.get_columns(
                    table_name, self._sql_database._schema
                )
            ):
                column_value_sample = [row[i] for row in eval(data_sample)]
                if col["name"] == table_pk_col:
                    table_col_list.append(
                        {
                            "table": table_name,
                            "column": col["name"],
                            "type": str(col["type"]),
                            "comment": col.get("comment"),
                            "primary_key": True,
                            "foreign_key": False,
                            "foreign_key_referred_table": None,
                            "data_sample": column_value_sample,
                            "extra_description": None,
                        }
                    )
                else:
                    table_col_list.append(
                        {
                            "table": table_name,
                            "column": col["name"],
                            "type": str(col["type"]),
                            "comment": col.get("comment"),
                            "primary_key": False,
                            "foreign_key": False,
                            "foreign_key_referred_table": None,
                            "data_sample": column_value_sample,
                            "extra_description": None,
                        }
                    )
            for foreign_key in self._sql_database._inspector.get_foreign_keys(
                table_name, self._sql_database._schema
            ):
                table_foreign_key = {
                    "table": table_name,
                    "column": foreign_key["constrained_columns"][0],
                    "foreign_key": True,
                    "foreign_key_referred_table": foreign_key["referred_table"],
                }
                table_foreign_key_list.append(table_foreign_key)

        # 处理table之间的foreign key一致性
        for table_foreign_key in table_foreign_key_list:
            for item in table_col_list:
                if (
                    item["table"] == table_foreign_key["table"]
                    and item["column"] == table_foreign_key["column"]
                ):
                    item.update(table_foreign_key)
                if (
                    item["table"] == table_foreign_key["foreign_key_referred_table"]
                    and item["column"] == table_foreign_key["column"]
                ):
                    item.update(
                        {
                            "foreign_key": True,
                            "foreign_key_referred_table": table_foreign_key["table"],
                        }
                    )

        structured_table_description = {
            "db_overview": None,
            "table_column_info": table_col_list,
            "table_opt_context": table_opt_context_list,
        }
        structured_table_description_str = json.dumps(
            structured_table_description, indent=4, ensure_ascii=False
        )
        logger.info("structured_table_description generated.")

        # 保存为txt文件
        if file_path is None:
            file_path = DEFAULT_DESC_FILE_PATH
        self._save_to_txt(structured_table_description_str, file_path)
        logger.info(f"structured_table_description saved to: {file_path}")

        return structured_table_description_str

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

    def _get_data_sample(self, table: str, sample_n: int = 3, seed: int = 2024) -> str:
        """对table随机采样"""
        if self._dialect == "mysql":
            # MySQL 使用 RAND(seed) 函数
            sql_str = f"SELECT * FROM {table} ORDER BY RAND({seed}) LIMIT {sample_n};"
        elif self._dialect == "sqlite":
            # SQLite 可以使用 RANDOM() 函数，但没有直接的种子设置
            sql_str = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {sample_n};"
        elif self._dialect == "postgresql":
            # PostgreSQL 可以使用 SETSEED() 函数设置随机种子
            set_seed_query = f"SELECT setseed({seed});"
            table_sample, _ = self._sql_database.run_sql(set_seed_query)
            sql_str = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {sample_n};"
        # if self._dialect in ("sqlite", "postgresql"):
        #     sql_str = f"Select * FROM {table} ORDER BY RANDOM() LIMIT {sample_n};"
        table_sample, _ = self._sql_database.run_sql(sql_str)

        return table_sample

    def get_enhanced_table_description(self, file_path: Optional[str] = None):
        """
        利用LLM总结table, 以及各个table/column的extra description
        """
        structured_table_description_str = self.get_structured_table_description(
            QueryBundle("")
        )
        table_col_list = json.loads(structured_table_description_str)[
            "table_column_info"
        ]
        tabel_opt_context_list = json.loads(structured_table_description_str)[
            "table_opt_context"
        ]
        table_col_df = pd.DataFrame(table_col_list)
        table_col_str = str(table_col_df)
        table_opt_context_str = str(tabel_opt_context_list)

        sllm = self._llm.as_structured_llm(output_cls=AnalysisOutput)
        output_summary = sllm.predict(
            prompt=self._db_summary_prompt,
            db_name=self._dbname,
            db_table_info=table_col_str,
            table_opt_context=table_opt_context_str,
        )

        # 与table_col_df拼接
        output_summary_dict = json.loads(output_summary)
        table_col_df_with_extra_desc = pd.merge(
            table_col_df.drop(columns="extra_description"),
            pd.DataFrame(output_summary_dict["table_column_info"]),
            on=["table", "column"],
            how="left",
        )
        # table_column_analysis更新llm信息
        output_summary_dict["table_column_info"] = table_col_df_with_extra_desc.to_dict(
            orient="records"
        )
        output_summary_dict["table_opt_context"] = []
        output_summary_str = json.dumps(
            output_summary_dict, indent=4, ensure_ascii=False
        )

        logger.info("llm enhanced db description generated.")

        # 保存为txt文件
        if file_path is None:
            file_path = DEFAULT_DESC_FILE_PATH
        self._save_to_txt(output_summary_str, file_path)
        logger.info(f"llm enhanced db description saved to: {file_path}")

        return output_summary_str

    async def aget_enhanced_table_description(self, file_path: Optional[str] = None):
        """利用LLM总结db info, 和 各个table/column的extra description"""
        table_col_list, tabel_opt_context_list = self.get_structured_table_description(
            QueryBundle("")
        )
        table_col_df = pd.DataFrame(table_col_list)
        table_info_df_str = str(table_col_df)
        table_opt_context_str = str(tabel_opt_context_list)

        sllm = self._llm.as_structured_llm(output_cls=AnalysisOutput)
        output_summary = sllm.apredict(
            prompt=self._db_summary_prompt,
            db_name=self._dbname,
            db_table_info=table_info_df_str,
            table_opt_context=table_opt_context_str,
        )

        # 与table_col_df拼接
        output_summary_dict = json.loads(output_summary)
        table_col_df_with_extra_desc = pd.merge(
            table_col_df.drop(columns="extra_description"),
            pd.DataFrame(output_summary_dict["table_column_info"]),
            on=["table", "column"],
            how="left",
        )
        # table_column_analysis更新llm信息
        output_summary_dict["table_column_info"] = table_col_df_with_extra_desc.to_dict(
            orient="records"
        )
        output_summary_dict["table_opt_context"] = []
        output_summary_str = json.dumps(
            output_summary_dict, indent=4, ensure_ascii=False
        )

        logger.info("async llm enhanced db description finished")

        # 保存为txt文件
        if file_path is None:
            file_path = DEFAULT_DESC_FILE_PATH
        self._save_to_txt(output_summary_str, file_path)

        logger.info(f"async llm enhanced db description saved to: {file_path}")

        return output_summary_str

    def _save_to_txt(self, content: str, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        return

    def collect_history(
        self, db_query_history: Any, file_path: Optional[str] = None
    ) -> None:
        """collect db query history and save it into txt file"""
        if file_path is None:
            file_path = DEFAULT_HISTORY_FILE_PATH
        db_query_history_str = str(db_query_history)  # TODO: some query_history parser
        self._save_to_txt(db_query_history_str, file_path)
        logger.info(f"db_query_history saved to: {file_path}")
        return

    # TODO
    def knowledge_index(
        self,
    ):
        """
        description 切分chunk，embedding，store，index
        history待处理
        db_value待处理
        """
        pass


class TableColumnDesc(BaseModel):
    """Data model for TableColumnDesc"""

    table: str = Field(description="表名")
    column: str = Field(description="字段名")
    extra_description: str = Field(description="字段的描述，包括专业术语解释，时间格式解释等")


class AnalysisOutput(BaseModel):
    """Data model for AnalysisOutput."""

    db_overview: str = Field(description="数据内容分析总结")
    table_column_info: List[TableColumnDesc] = Field(
        description="每个字段的补充描述, 禁止生成参考信息以外的内容"
    )
