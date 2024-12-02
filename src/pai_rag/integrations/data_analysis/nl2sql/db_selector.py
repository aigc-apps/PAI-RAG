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
    get_schema_desc4llm,
    count_total_columns,
    get_target_info,
    extract_subset_from_description,
)
from pai_rag.integrations.data_analysis.nl2sql.db_utils.constants import (
    DEFAULT_DB_DESCRIPTION_PATH,
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
        # 如果没有传入db_description_dict, 则从文件读取
        db_description_dict = get_target_info(
            self._db_description_path, db_description_dict, "description"
        )
        total_columns = count_total_columns(db_description_dict)
        schema_description_str = get_schema_desc4llm(db_description_dict)

        sllm = self._llm.as_structured_llm(output_cls=SchemaSelection)
        selected_output_str = sllm.predict(
            prompt=self._db_schema_select_prompt,
            nl_query=nl_query.query_str,
            db_schema=schema_description_str,
        )
        logger.info(f"selected_output_str: \n{selected_output_str}\n")
        # 解析筛选
        selected_db_description_dict = self._filter_selection(
            selected_output_str, db_description_dict
        )
        selected_total_columns = count_total_columns(selected_db_description_dict)
        logger.info(
            f"Description selected, number from {total_columns} to {selected_total_columns}."
        )

        return selected_db_description_dict

    async def aselect_schema(
        self,
        nl_query: QueryType,
        db_description_dict: Optional[Dict] = None,
    ) -> Dict:
        """
        选择相关的表和列
        """
        # 如果没有传入db_description_dict, 则从文件读取
        db_description_dict = get_target_info(
            self._db_description_path, db_description_dict, "description"
        )
        total_columns = count_total_columns(db_description_dict)
        schema_description_str = get_schema_desc4llm(db_description_dict)

        sllm = self._llm.as_structured_llm(output_cls=SchemaSelection)
        selected_output_str = await sllm.apredict(
            prompt=self._db_schema_select_prompt,
            nl_query=nl_query.query_str,
            db_schema=schema_description_str,
        )
        logger.info(f"selected_output_str: \n{selected_output_str}\n")
        # 解析筛选
        selected_db_description_dict = self._filter_selection(
            selected_output_str, db_description_dict
        )
        selected_total_columns = count_total_columns(selected_db_description_dict)
        logger.info(
            f"Description selected, number from {total_columns} to {selected_total_columns}."
        )

        return selected_db_description_dict

    def _filter_selection(
        self, selected_output_str: str, db_description_dict: Dict
    ) -> Dict:
        """从schema selection的结果筛选db_description"""
        selected_output_dict = json.loads(selected_output_str)
        selected_table_col_dict = {}
        if len(selected_output_dict) > 0:
            for item in selected_output_dict["selected_info"]:
                key = (item["table"], item["column"])
                if key not in selected_table_col_dict:
                    selected_table_col_dict[key] = []
            logger.info(
                f"selected_table_col_dict: {len(selected_table_col_dict)},\n {selected_table_col_dict}"
            )
        else:
            logger.info("Empty selected_output_dict")

        # 过滤db_description_dict
        filterd_db_description_dict = extract_subset_from_description(
            selected_table_col_dict, db_description_dict
        )

        return filterd_db_description_dict

    def select_table(self):
        pass

    def select_column(self):
        pass


class ColumnSelection(BaseModel):
    table: str = Field(description="表名")
    column: str = Field(description="字段名")


class SchemaSelection(BaseModel):
    selected_info: List[ColumnSelection] = Field(description="筛选出的表名和字段名，通常包含多个")
