import os
import json
from typing import Any, Callable, Dict, List, Optional, Union, cast
from decimal import Decimal
from loguru import logger
import datetime
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

from pai_rag.integrations.data_analysis.nl2sql.db_utils.nl2sql_utils import (
    get_schema_desc4llm,
)
from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_DB_SUMMARY_PROMPT,
)
from pai_rag.integrations.data_analysis.nl2sql.db_utils.constants import (
    DEFAULT_DB_DESCRIPTION_PATH,
)


class DBDescriptor(PromptMixin):
    """
    收集数据库结构信息(表名、列名、采样数据等): DDL + data sample + opt description
    基于结构信息利用LLM进行信息总结补充
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
            db_description_file_path or DEFAULT_DB_DESCRIPTION_PATH
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

    def get_structured_db_description(
        self, query_bundle: QueryBundle, file_path: Optional[str] = None
    ) -> Dict:
        """
        Get structured table schema with data samples + optional context description from webui
        """
        table_schema_objs = self._get_tables(
            query_bundle.query_str
        )  # get a list of SQLTableSchema, e.g. [SQLTableSchema(table_name='has_pet', context_str=None),]
        table_info_list = []
        table_foreign_key_list = []

        for table_schema_obj in table_schema_objs:
            table_name = table_schema_obj.table_name
            comment_from_db = self._sql_database._inspector.get_table_comment(
                table_name, schema=self._sql_database._schema
            )["text"]
            additional_desc = table_schema_obj.context_str
            # get table description
            table_comment = self._merge_comment_and_desc(
                comment_from_db, additional_desc
            )
            # get table data samples
            data_sample = self._get_data_sample(table_name)
            # get table primary key
            table_pk_col = self._get_table_primary_key(table_name)
            # get foreign keys
            table_fks = self._get_table_foreign_keys(table_name)
            table_foreign_key_list.extend(table_fks)
            # get column info
            column_info_list = []
            for i, col in enumerate(
                self._sql_database._inspector.get_columns(
                    table_name, self._sql_database._schema
                )
            ):
                # print("col:", col, "data_sample:", data_sample)
                column_value_sample = [row[i] for row in data_sample]
                # collect and structure table schema with data samples
                column_info_list.append(
                    {
                        "column_name": col["name"],
                        "column_type": str(col["type"]),
                        "column_comment": col.get("comment"),
                        "primary_key": col["name"] == table_pk_col,
                        "foreign_key": False,
                        "foreign_key_referred_table": None,
                        "column_value_sample": column_value_sample,
                        "column_description": None,
                    }
                )

            table_info_list.append(
                {
                    "table_name": table_name,
                    "table_comment": table_comment,
                    "table_description": None,
                    "column_info": column_info_list,
                }
            )

        # 处理table之间的foreign key一致性
        table_info_list = self._keep_foreign_keys_consistency(
            table_foreign_key_list, table_info_list
        )

        structured_db_description_dict = {
            "db_overview": None,
            "table_info": table_info_list,
        }

        logger.info("structured_db_description generated.")

        # 保存为json文件
        if file_path is None:
            file_path = self._db_description_file_path
        self._save_to_json(structured_db_description_dict, file_path)
        logger.info(f"structured_db_description saved to: {file_path}")

        return structured_db_description_dict

    def get_enhanced_db_description(self, file_path: Optional[str] = None) -> None:
        """
        利用LLM总结table, 以及各个table/column的description
        """

        db_description_dict = self.get_structured_db_description(QueryBundle(""))
        schema_description_str = get_schema_desc4llm(db_description_dict)
        logger.info(f"schema description for llm: \n {schema_description_str}")

        sllm = self._llm.as_structured_llm(output_cls=AnalysisOutput)
        output_summary_str = sllm.predict(
            prompt=self._db_summary_prompt,
            db_name=self._dbname,
            db_schema=schema_description_str,
        )

        saved_file_path = self._postprocess_enhanced_description(
            db_description_dict, output_summary_str, file_path
        )
        logger.info(
            f"llm enhanced db description generated and saved to: {saved_file_path}"
        )

        return

    async def aget_enhanced_db_description(
        self, file_path: Optional[str] = None
    ) -> None:
        """
        利用LLM总结table, 以及各个table/column的description
        """
        db_description_dict = self.get_structured_db_description(QueryBundle(""))
        schema_description_str = get_schema_desc4llm(db_description_dict)
        logger.info(f"schema description for llm: \n {schema_description_str}")

        sllm = self._llm.as_structured_llm(output_cls=AnalysisOutput)
        output_summary_str = await sllm.apredict(
            prompt=self._db_summary_prompt,
            db_name=self._dbname,
            db_schema=schema_description_str,
        )

        saved_file_path = self._postprocess_enhanced_description(
            db_description_dict, output_summary_str, file_path
        )
        logger.info(
            f"async llm enhanced db description generated and saved to: {saved_file_path}"
        )

        return

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

    def _get_data_sample(self, table: str, sample_n: int = 3, seed: int = 2024) -> List:
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

        # 转换 Decimal 对象为 float，datetime 对象为字符串
        converted_table_sample = self._convert_data_sample_format(eval(table_sample))
        # print("converted_table_sample:", converted_table_sample)

        return converted_table_sample

    def _get_table_primary_key(self, table_name: str) -> str:
        table_pk = self._sql_database._inspector.get_pk_constraint(
            table_name, self._sql_database._schema
        )  # get primary key
        if len(table_pk["constrained_columns"]) > 0:
            table_pk_col = table_pk["constrained_columns"][0]
        else:
            table_pk_col = None

        return table_pk_col

    def _get_table_foreign_keys(self, table_name: str) -> List:
        table_fks = []
        for foreign_key in self._sql_database._inspector.get_foreign_keys(
            table_name, self._sql_database._schema
        ):
            table_foreign_key = {
                "table_name": table_name,
                "column_name": foreign_key["constrained_columns"][0],
                "foreign_key": True,
                "foreign_key_referred_table": foreign_key["referred_table"],
            }
            table_fks.append(table_foreign_key)

        return table_fks

    def _merge_comment_and_desc(self, comment_from_db: str, additional_desc: str):
        target_comment = [
            value for value in [comment_from_db, additional_desc] if value is not None
        ]
        if len(target_comment) > 0:
            return ", ".join(target_comment)
        else:
            return None

    def _keep_foreign_keys_consistency(self, table_foreign_key_list, table_info_list):
        # 处理table之间的foreign key一致性
        for table_foreign_key in table_foreign_key_list:
            for table_item in table_info_list:
                for column_item in table_item["column_info"]:
                    if (
                        table_item["table_name"] == table_foreign_key["table_name"]
                        and column_item["column_name"]
                        == table_foreign_key["column_name"]
                    ):
                        column_item.update(
                            {
                                "foreign_key": True,
                                "foreign_key_referred_table": table_foreign_key[
                                    "foreign_key_referred_table"
                                ],
                            }
                        )
                    if (
                        table_item["table_name"]
                        == table_foreign_key["foreign_key_referred_table"]
                        and column_item["column_name"]
                        == table_foreign_key["column_name"]
                    ):
                        column_item.update(
                            {
                                "foreign_key": True,
                                "foreign_key_referred_table": table_foreign_key[
                                    "table_name"
                                ],
                            }
                        )
        return table_info_list

    def _convert_data_sample_format(self, data):
        """递归地将数据中的特殊类型转换为常规类型"""
        if isinstance(data, list):
            return [self._convert_data_sample_format(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._convert_data_sample_format(item) for item in data)
        elif isinstance(data, dict):
            return {
                key: self._convert_data_sample_format(value)
                for key, value in data.items()
            }
        elif isinstance(data, Decimal):
            return float(data)
        elif isinstance(data, datetime.datetime):
            return data.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(data, datetime.date):
            return data.strftime("%Y-%m-%d")
        elif isinstance(data, datetime.time):
            return data.strftime("%H:%M:%S")
        elif isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")  # 将 bytes 转换为 str
        else:
            return data

    def _postprocess_enhanced_description(
        self,
        db_description_dict: Dict,
        output_summary_str: str,
        file_path: Optional[str] = None,
    ) -> None:
        """将LLM生成的description合并到table_info_df & column_info_df中 并保存"""
        output_summary_dict = json.loads(output_summary_str)
        db_description_dict["db_overview"] = output_summary_dict["db_overview"]
        for table in db_description_dict["table_info"]:
            table_name = table["table_name"]
            output_summary_table = [
                item
                for item in output_summary_dict["table_info"]
                if item["table"] == table_name
            ][0]
            table["table_description"] = output_summary_table["description"]
            for column in table["column_info"]:
                column_name = column["column_name"]
                output_summary_column = [
                    item
                    for item in output_summary_table["column_info"]
                    if item["column"] == column_name
                ][0]
                column["column_description"] = output_summary_column["description"]

        # 保存为json文件
        if file_path is None:
            file_path = self._db_description_file_path
        self._save_to_json(db_description_dict, file_path)

        return file_path

    def _save_to_json(self, content: Dict, file_path: str) -> None:
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
                json.dump(content, file, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Error saving content to file: {e}")
        return


class ColumnDesc(BaseModel):
    """Data model for ColumnDesc"""

    column: str = Field(description="字段名")
    description: str = Field(description="字段的描述，包括专业术语解释，时间格式解释等，无需重复字段类型等已知信息")


class TableDesc(BaseModel):
    """Data model for TableDesc."""

    table: str = Field(description="表名")
    description: str = Field(description="表的概述")
    column_info: List[ColumnDesc] = Field(description="每个字段的补充描述")


class AnalysisOutput(BaseModel):
    """Data model for AnalysisOutput."""

    db_overview: str = Field(description="数据库信息分析总结")
    table_info: List[TableDesc] = Field(description="每个表的补充描述")
