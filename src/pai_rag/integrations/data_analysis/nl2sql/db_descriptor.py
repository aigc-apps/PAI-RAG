import os
import logging
import json
from typing import Any, Callable, Dict, List, Optional, Union, cast

import pandas as pd
from pydantic.v1 import BaseModel, Field
from sqlalchemy import Table

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
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

from pai_rag.integrations.data_analysis.nl2sql.nl2sql_utils import (
    generate_schema_description,
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
        db_name: str,
        context_query_kwargs: Optional[dict] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
        context_str_prefix: Optional[str] = None,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        service_context: Optional[ServiceContext] = None,
        db_summary_prompt: Optional[BasePromptTemplate] = None,
        db_description_file_path: Optional[str] = None,
    ) -> None:
        self._sql_database = sql_database
        self._dbname = db_name
        self._tables = list(sql_database._usable_tables)
        self._dialect = sql_database.dialect
        self._get_tables = self._load_get_tables_fn(
            sql_database, self._tables, context_query_kwargs, table_retriever
        )
        self._context_str_prefix = context_str_prefix
        self._llm = llm or Settings.llm
        self._embed_model = embed_model or Settings.embed_model
        self._db_summary_prompt = db_summary_prompt or DEFAULT_DB_SUMMARY_PROMPT
        self._db_description_file_path = (
            db_description_file_path or DEFAULT_DESC_FILE_PATH
        )

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
    ) -> Dict:
        """
        Get structured table schema with data samples + optional context description from webui
        """
        table_schema_objs = self._get_tables(
            query_bundle.query_str
        )  # get a list of SQLTableSchema, e.g. [SQLTableSchema(table_name='has_pet', context_str=None),]
        table_info_list = []
        column_info_list = []
        table_foreign_key_list = []

        for table_schema_obj in table_schema_objs:
            table_name = table_schema_obj.table_name
            table_comment = self._sql_database._inspector.get_table_comment(
                table_name, schema=self._sql_database._schema
            )["text"]
            # collect table level info
            table_info_list.append(
                {
                    "table": table_name,
                    "comment": table_comment,
                    "description": table_schema_obj.context_str,
                    "overview": None,
                }
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
                    column_info_list.append(
                        {
                            "table": table_name,
                            "column": col["name"],
                            "type": str(col["type"]),
                            "comment": col.get("comment"),
                            "primary_key": True,
                            "foreign_key": False,
                            "foreign_key_referred_table": None,
                            "value_sample": column_value_sample,
                            "description": None,
                        }
                    )
                else:
                    column_info_list.append(
                        {
                            "table": table_name,
                            "column": col["name"],
                            "type": str(col["type"]),
                            "comment": col.get("comment"),
                            "primary_key": False,
                            "foreign_key": False,
                            "foreign_key_referred_table": None,
                            "value_sample": column_value_sample,
                            "description": None,
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
            for item in column_info_list:
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

        structured_table_description_dict = {
            "db_overview": None,
            "table_info": table_info_list,
            "column_info": column_info_list,
        }
        structured_table_description_str = json.dumps(
            structured_table_description_dict, indent=4, ensure_ascii=False
        )
        logger.info("structured_table_description generated.")

        # 保存为txt文件
        if file_path is None:
            file_path = self._db_description_file_path
        self._save_to_txt(structured_table_description_str, file_path)
        logger.info(f"structured_table_description saved to: {file_path}")

        return structured_table_description_dict

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

    # def generate_schema_description(self, file_path: Optional[str] = None) -> str:
    #     """"
    #     基于结构化的数据库信息，生成适合llm的数据库描述，包括表结构、表描述、列描述等
    #     """
    #     if file_path is None:
    #         file_path = self._db_description_file_path
    #     with open(file_path, 'r') as f:
    #         structured_table_description_dict = json.load(f)

    #     table_info_list = structured_table_description_dict[
    #         "table_info"
    #     ]
    #     column_info_list = structured_table_description_dict[
    #         "column_info"
    #     ]
    #     table_info_df = pd.DataFrame(table_info_list)
    #     column_info_df = pd.DataFrame(column_info_list)

    #     # 生成所有表的描述
    #     all_table_descriptions = []
    #     for table_name in table_info_df['table']:
    #         all_table_descriptions.append(self._generate_single_table_description(table_name, table_info_df, column_info_df))

    #     # 将所有表的描述合并成一个字符串
    #     schema_description_str = "\n".join(all_table_descriptions)

    #     return schema_description_str, table_info_df, column_info_df

    # def _generate_single_table_description(self, table_name, table_info_df, column_info_df) -> str:
    #     """
    #     基于单表的结构化信息，生成适合llm的数据库描述，包括表结构、表描述、列描述等
    #     """
    #     table_row = table_info_df[table_info_df["table"] == table_name].iloc[0]
    #     columns = column_info_df[column_info_df["table"] == table_name]

    #     table_desc = f"Table {table_name} has columns: "
    #     for _, column in columns.iterrows():
    #         table_desc += f""" {column["column"]} ({column["type"]})"""
    #         if column["primary_key"]:
    #             table_desc += ", Primary Key"
    #         if column["foreign_key"]:
    #             table_desc += f""", Foreign Key, Referred Table: {column["foreign_key_referred_table"]}"""
    #         table_desc += f""", with Value Sample: {column["value_sample"]}"""
    #         if column["comment"] or column["description"]:
    #             table_desc += f""", with Description: {column["comment"] or ""}, {column['description'] or ""};"""
    #         else:
    #             table_desc += ";"
    #     if table_row["comment"] or table_row["description"] or table_row["overview"]:
    #         table_desc += f""" with Table Description: {table_row["comment"] or ""}, {table_row["description"] or ""}, {table_row["overview"] or ""}.\n"""
    #     else:
    #         table_desc += f".\n "

    #     return table_desc

    def _merge_llm_info(
        self, table_info_df, column_info_df, output_summary_dict: Dict
    ) -> str:
        """
        将LLM生成的extra_description合并到table_info_df & column_info_df中
        """
        table_info_df_with_overview = pd.merge(
            table_info_df.drop(columns=["overview"]),
            pd.DataFrame(output_summary_dict["table_info"]),
            on=["table"],
            how="left",
        )
        column_info_with_desc = pd.merge(
            column_info_df.drop(columns=["description"]),
            pd.DataFrame(output_summary_dict["column_info"]),
            on=["table", "column"],
            how="left",
        )

        output_summary_dict["table_info"] = table_info_df_with_overview.to_dict(
            orient="records"
        )
        output_summary_dict["column_info"] = column_info_with_desc.to_dict(
            orient="records"
        )
        output_summary_str = json.dumps(
            output_summary_dict, indent=4, ensure_ascii=False
        )

        return output_summary_str

    def _save_to_txt(self, content: str, file_path: str) -> None:
        """
        将内容保存到指定的文本文件中
        :param content: 要保存的内容
        :param file_path: 文件路径
        """
        try:
            directory = os.path.dirname(file_path)
            # 检查文件夹是否存在
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
        except Exception as e:
            logger.error(f"Error saving content to file: {e}")
        return

    def get_enhanced_table_description(self, file_path: Optional[str] = None):
        """
        利用LLM总结table, 以及各个table/column的description
        """
        # schema_description_str, table_info_df, column_info_df = self.generate_schema_description()
        structured_table_description_dict = self.get_structured_table_description(
            QueryBundle("")
        )
        structured_db_description_str = json.dumps(structured_table_description_dict)

        # if file_path is None:
        #     file_path = self._db_description_file_path
        # with open(file_path, "r") as f:
        #     structured_db_description_str = f.read()

        (
            schema_description_str,
            table_info_df,
            column_info_df,
        ) = generate_schema_description(structured_db_description_str)
        logger.info(f"schema description for llm: \n {schema_description_str}")

        sllm = self._llm.as_structured_llm(output_cls=AnalysisOutput)
        output_summary_str = sllm.predict(
            prompt=self._db_summary_prompt,
            db_name=self._dbname,
            db_schema=schema_description_str,
        )
        output_summary_dict = json.loads(output_summary_str)

        # 更新table_info_df & column_info_df
        output_description_str = self._merge_llm_info(
            table_info_df, column_info_df, output_summary_dict
        )
        logger.info("llm enhanced db description generated.")

        # 保存为txt文件
        if file_path is None:
            file_path = self._db_description_file_path
        self._save_to_txt(output_description_str, file_path)
        logger.info(f"llm enhanced db description saved to: {file_path}")

        return output_description_str

    async def aget_enhanced_table_description(self, file_path: Optional[str] = None):
        """
        利用LLM总结table, 以及各个table/column的description
        """
        structured_table_description_dict = self.get_structured_table_description(
            QueryBundle("")
        )
        structured_db_description_str = json.dumps(structured_table_description_dict)

        # if file_path is None:
        #     file_path = self._db_description_file_path
        # with open(file_path, "r") as f:
        #     structured_db_description_str = f.read()

        (
            schema_description_str,
            table_info_df,
            column_info_df,
        ) = generate_schema_description(structured_db_description_str)
        logger.info(f"schema description for llm: \n {schema_description_str}")

        sllm = self._llm.as_structured_llm(output_cls=AnalysisOutput)
        output_summary_str = await sllm.predict(
            prompt=self._db_summary_prompt,
            db_name=self._dbname,
            db_schema=schema_description_str,
        )
        output_summary_dict = json.loads(output_summary_str)

        # 更新table_info_df & column_info_df
        output_description_str = self._merge_llm_info(
            table_info_df, column_info_df, output_summary_dict
        )
        logger.info("async llm enhanced db description generated.")

        # 保存为txt文件
        if file_path is None:
            file_path = self._db_description_file_path
        self._save_to_txt(output_description_str, file_path)
        logger.info(f"async llm enhanced db description saved to: {file_path}")

        return output_description_str

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
    def embed_db_context(
        self,
        context: str,
    ):
        """
        处理三类db_info的embedding
        description 切分chunk，embedding，store，index
        history
        db_value
        """

        pass


class ColumnDesc(BaseModel):
    """Data model for ColumnDesc"""

    table: str = Field(description="表名")
    column: str = Field(description="字段名")
    description: str = Field(description="字段的描述，包括专业术语解释，时间格式解释等")


class TableDesc(BaseModel):
    """Data model for TableDesc."""

    table: str = Field(description="表名")
    overview: str = Field(description="表的概述")


class AnalysisOutput(BaseModel):
    """Data model for AnalysisOutput."""

    db_overview: str = Field(description="数据库信息分析总结")
    table_info: List[TableDesc] = Field(description="每个表的补充描述")
    column_info: List[ColumnDesc] = Field(description="每个字段的补充描述, 无需重复字段类型等已知信息")
