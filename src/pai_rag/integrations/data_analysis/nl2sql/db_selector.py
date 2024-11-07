import os
import logging
import json
from pydantic.v1 import BaseModel, Field
from typing import List, Optional

from llama_index.core import BasePromptTemplate
from llama_index.core.schema import QueryBundle
from llama_index.core.llms.llm import LLM
from llama_index.core import Settings

from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_DB_SCHEMA_SELECT_PROMPT,
)
from pai_rag.integrations.data_analysis.nl2sql.nl2sql_utils import (
    generate_schema_description,
)

logger = logging.getLogger(__name__)

DEFAULT_DESC_FILE_PATH = "./db_structured_description.txt"


class DBSelector:
    """
    基于descriptor或pretriever返回的结果，descriptor必须, pretriever可选
    通过llm chain of thought 进一步选择table和column，输出为缩小的db_description
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        db_schema_select_prompt: Optional[BasePromptTemplate] = None,
        db_description_file_path: Optional[str] = None,
    ) -> None:
        self._llm = llm or Settings.llm
        self._db_schema_select_prompt = (
            db_schema_select_prompt or DEFAULT_DB_SCHEMA_SELECT_PROMPT
        )
        self._db_description_file_path = (
            db_description_file_path or DEFAULT_DESC_FILE_PATH
        )

    def _parse_selection(
        self, selected_output_str: str, db_description_str: str
    ) -> str:
        """
        从schema selection中解析出table和column, 然后条件筛选description, 返回筛选后的db_description
        """
        selected_output_dict = json.loads(selected_output_str)
        db_description_dict = json.loads(db_description_str)
        # filter column_info
        selected_db_description_list = []
        for item in db_description_dict["column_info"]:
            if {
                "table": item["table"],
                "column": item["column"],
            } in selected_output_dict["column_info"]:
                selected_db_description_list.append(item)
            if item["primary_key"] is True or item["foreign_key"] is True:  # 保留主键和外键
                selected_db_description_list.append(item)
        # update with selected_db_description_list
        db_description_dict["column_info"] = selected_db_description_list
        selected_db_description_str = json.dumps(
            db_description_dict, indent=4, ensure_ascii=False
        )

        return selected_db_description_str

    def select_schema(
        self,
        nl_query: QueryBundle,
        db_description_str: Optional[str] = None,  # 支持额外传入
    ) -> str:
        """
        选择相关的表和列
        """
        # 如果online流程没有选择preretriever直接使用selector，则一般从db_structured_description.txt中读取
        if db_description_str is None:
            if os.path.exists(self._db_description_file_path):
                file_path = self._db_description_file_path
            else:
                raise ValueError(
                    f"db_description_file_path: {self._db_description_file_path} does not exist"
                )

            with open(file_path, "r") as f:
                db_description_str = f.read()

        schema_description_str, _, _ = generate_schema_description(db_description_str)

        sllm = self._llm.as_structured_llm(output_cls=SchemaSelection)
        selected_output_str = sllm.predict(
            prompt=self._db_schema_select_prompt,
            nl_query=nl_query.query_str,
            db_schema=schema_description_str,  # 提供给llm的都是
        )
        logger.info(f"selected_output_str: \n{selected_output_str}\n")
        # 解析筛选
        selected_description_str = self._parse_selection(
            selected_output_str, db_description_str
        )
        logger.info(f"selected_description_str: \n{selected_description_str}\n")

        return selected_description_str

    async def aselect_schema(
        self,
        nl_query: QueryBundle,
        db_description_str: Optional[str] = None,  # 支持额外传入
    ) -> str:
        """
        选择相关的表和列
        """
        # 如果online流程没有选择preretriever直接使用selector，则一般从db_structured_description.txt中读取
        if db_description_str is None:
            if os.path.exists(self._db_description_file_path):
                file_path = self._db_description_file_path
            else:
                raise ValueError(
                    f"db_description_file_path: {self._db_description_file_path} does not exist"
                )

            with open(file_path, "r") as f:
                db_description_str = f.read()

        schema_description_str, _, _ = generate_schema_description(db_description_str)

        sllm = self._llm.as_structured_llm(output_cls=SchemaSelection)
        selected_output_str = await sllm.apredict(
            prompt=self._db_schema_select_prompt,
            nl_query=nl_query.query_str,
            db_schema=schema_description_str,  # 提供给llm的都是
        )
        logger.info(f"selected_output_str: \n{selected_output_str}\n")
        # 解析筛选
        selected_description_str = self._parse_selection(
            selected_output_str, db_description_str
        )
        logger.info(f"selected_description_str: \n{selected_description_str}\n")

        return selected_description_str

    def select_table(self):
        pass

    def select_column(self):
        pass


class ColumnSelection(BaseModel):
    table: str = Field(description="表名")
    column: str = Field(description="字段名")


class SchemaSelection(BaseModel):
    column_info: List[ColumnSelection] = Field(description="筛选出的表名和字段名，通常包含多个")
