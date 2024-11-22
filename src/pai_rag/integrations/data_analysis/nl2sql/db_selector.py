import os
from loguru import logger
import json
from pydantic.v1 import BaseModel, Field
from typing import List, Optional, Dict

from llama_index.core import BasePromptTemplate
from llama_index.core.schema import QueryType
from llama_index.core.llms.llm import LLM
from llama_index.core import Settings

from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_DB_SCHEMA_SELECT_PROMPT,
)
from pai_rag.integrations.data_analysis.nl2sql.db_utils.nl2sql_utils import (
    generate_schema_description,
)


DEFAULT_DB_DESCRIPTION_PATH = (
    "./localdata/data_analysis/nl2sql/db_structured_description.json"
)


class DBSelector:
    """
    基于descriptor或pretriever返回的结果，descriptor必须, pretriever可选
    通过llm chain of thought 进一步选择table和column，输出为缩小的db_description
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        db_schema_select_prompt: Optional[BasePromptTemplate] = None,
        db_description_path: Optional[str] = None,
    ) -> None:
        self._llm = llm or Settings.llm
        self._db_schema_select_prompt = (
            db_schema_select_prompt or DEFAULT_DB_SCHEMA_SELECT_PROMPT
        )
        self._db_description_path = db_description_path or DEFAULT_DB_DESCRIPTION_PATH

    def select_schema(
        self,
        nl_query: QueryType,
        db_description_dict: Optional[Dict] = None,
    ) -> Dict:
        """
        选择相关的表和列
        """
        # if isinstance(nl_query, str):
        #     nl_query = QueryBundle(nl_query)
        # else:
        #     nl_query = nl_query

        # 如果没有传入db_description_dict, 则从文件读取
        db_description_dict = self._get_schema_info(db_description_dict)
        schema_description_str, _, _ = generate_schema_description(db_description_dict)

        sllm = self._llm.as_structured_llm(output_cls=SchemaSelection)
        selected_output_str = sllm.predict(
            prompt=self._db_schema_select_prompt,
            nl_query=nl_query.query_str,
            db_schema=schema_description_str,  # 提供给llm的都是
        )
        logger.info(f"selected_output_str: \n{selected_output_str}\n")
        # 解析筛选
        selected_description_dict = self._parse_selection(
            selected_output_str, db_description_dict
        )
        logger.info(f"selected_description: \n{str(selected_description_dict)}\n")

        return selected_description_dict

    async def aselect_schema(
        self,
        nl_query: QueryType,
        db_description_dict: Optional[Dict] = None,  # 支持额外传入
    ) -> Dict:
        """
        选择相关的表和列
        """
        # if isinstance(nl_query, str):
        #     nl_query = QueryBundle(nl_query)
        # else:
        #     nl_query = nl_query

        # 如果没有传入db_description_dict, 则从文件读取
        db_description_dict = self._get_schema_info(db_description_dict)
        schema_description_str, _, _ = generate_schema_description(db_description_dict)

        sllm = self._llm.as_structured_llm(output_cls=SchemaSelection)
        selected_output_str = await sllm.apredict(
            prompt=self._db_schema_select_prompt,
            nl_query=nl_query.query_str,
            db_schema=schema_description_str,  # 提供给llm的都是
        )
        logger.info(f"selected_output_str: \n{selected_output_str}\n")
        # 解析筛选
        selected_description_dict = self._parse_selection(
            selected_output_str, db_description_dict
        )
        logger.info(f"selected_description_str: \n{str(selected_description_dict)}\n")

        return selected_description_dict

    def _get_schema_info(self, db_description_dict: Optional[Dict] = None) -> Dict:
        """get schema description info"""
        if db_description_dict is None:
            if os.path.exists(self._db_description_path):
                file_path = self._db_description_path
            else:
                raise ValueError(
                    f"db_description_file_path: {self._db_description_path} does not exist"
                )
            try:
                with open(file_path, "r") as f:
                    db_description_dict = json.load(f)
            except Exception as e:
                raise ValueError(
                    f"Load db_description_dict from {file_path} failed: {e}"
                )

        return db_description_dict

    def _parse_selection(
        self, selected_output_str: str, db_description_dict: Dict
    ) -> Dict:
        """
        从schema selection中解析出table和column, 然后条件筛选description, 返回筛选后的db_description
        """
        selected_output_dict = json.loads(selected_output_str)
        # filter column_info
        selected_db_description_list = []
        for item in db_description_dict["column_info"]:
            if {
                "table": item["table"],
                "column": item["column"],
            } in selected_output_dict["column_info"]:
                selected_db_description_list.append(item)
            if (item["primary_key"] is True or item["foreign_key"] is True) and (
                item not in selected_db_description_list
            ):  # 保留主键和外键
                selected_db_description_list.append(item)
        # update with selected_db_description_list
        db_description_dict["column_info"] = selected_db_description_list
        # selected_db_description_str = json.dumps(
        #     db_description_dict, indent=4, ensure_ascii=False
        # )

        return db_description_dict

    def select_table(self):
        pass

    def select_column(self):
        pass


class ColumnSelection(BaseModel):
    table: str = Field(description="表名")
    column: str = Field(description="字段名")


class SchemaSelection(BaseModel):
    column_info: List[ColumnSelection] = Field(description="筛选出的表名和字段名，通常包含多个")
