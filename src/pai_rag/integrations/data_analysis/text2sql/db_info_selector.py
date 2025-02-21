from abc import ABC, abstractmethod
from loguru import logger
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from llama_index.core import BasePromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import QueryType
from llama_index.core import Settings
from llama_index.core.program import LLMTextCompletionProgram

from pai_rag.integrations.data_analysis.text2sql.utils.prompts import (
    DEFAULT_DB_SCHEMA_SELECT_PROMPT,
)
from pai_rag.integrations.data_analysis.text2sql.utils.info_utils import (
    get_schema_desc,
    count_total_columns,
    extract_subset_from_description,
)


class DBInfoSelector(ABC):
    def __init__(
        self,
        llm: Optional[LLM] = None,
    ):
        self._llm = llm or Settings.llm

    @abstractmethod
    def select(self, query, db_info, hint: str = None):
        pass

    @abstractmethod
    async def aselect(self, query, db_info, hint: str = None):
        pass


class SchemaSelector(DBInfoSelector):
    def __init__(
        self,
        llm: Optional[LLM] = None,
        db_schema_select_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        super().__init__(llm)
        self._db_schema_select_prompt = (
            db_schema_select_prompt or DEFAULT_DB_SCHEMA_SELECT_PROMPT
        )

    def select(self, query: QueryType, db_info: Dict, hint: str = None):
        column_nums = count_total_columns(db_info)
        schema_description_str = get_schema_desc(db_info)

        # for llms that support FunctionCallingProgram
        # selected_output_obj = self._llm.structured_predict(
        #     output_cls=SchemaSelection,
        #     prompt=self._db_schema_select_prompt,
        #     llm_kwargs={
        #         "tool_choice": {
        #             "type": "function",
        #             "function": {"name": "SchemaSelection"},
        #         }
        #     },
        #     nl_query=query.query_str,
        #     hint=hint,
        #     db_schema=schema_description_str,
        # )

        # for llms that do not support FunctionCallingProgram
        textCompletion = LLMTextCompletionProgram.from_defaults(
            output_cls=SchemaSelection,
            llm=self._llm,
            prompt=self._db_schema_select_prompt,
        )
        selected_output_obj = textCompletion(
            nl_query=query.query_str, hint=hint, db_schema=schema_description_str
        )
        logger.info(f"selected_output_obj: \n{selected_output_obj}\n")
        # 解析筛选
        selected_db_description_dict = self._filter_selection(
            selected_output_obj, db_info
        )
        selected_column_nums = count_total_columns(selected_db_description_dict)
        logger.info(
            f"Description selected, number from {column_nums} to {selected_column_nums}."
        )

        return selected_db_description_dict

    async def aselect(self, query: QueryType, db_info: Dict, hint: str = None):
        column_nums = count_total_columns(db_info)
        schema_description_str = get_schema_desc(db_info)

        # for llms that support FunctionCallingProgram
        # selected_output_obj = await self._llm.astructured_predict(
        #     output_cls=SchemaSelection,
        #     prompt=self._db_schema_select_prompt,
        #     llm_kwargs={
        #         "tool_choice": {
        #             "type": "function",
        #             "function": {"name": "SchemaSelection"},
        #         }
        #     },
        #     nl_query=query.query_str,
        #     hint=hint,
        #     db_schema=schema_description_str,
        # )

        # for llms that do not support FunctionCallingProgram
        textCompletion = LLMTextCompletionProgram.from_defaults(
            output_cls=SchemaSelection,
            llm=self._llm,
            prompt=self._db_schema_select_prompt,
        )
        selected_output_obj = await textCompletion.acall(
            nl_query=query.query_str, hint=hint, db_schema=schema_description_str
        )
        logger.info(f"selected_output_str: \n{selected_output_obj}\n")
        # 解析筛选
        selected_db_description_dict = self._filter_selection(
            selected_output_obj, db_info
        )
        selected_column_nums = count_total_columns(selected_db_description_dict)
        logger.info(
            f"Description selected, number from {column_nums} to {selected_column_nums}."
        )

        return selected_db_description_dict

    def _filter_selection(
        self, selected_output_obj: str, db_description_dict: Dict
    ) -> Dict:
        """从schema selection的结果筛选db_description"""
        selected_output_list = selected_output_obj.selected_info
        selected_table_col_dict = {}
        if len(selected_output_list) > 0:
            for item in selected_output_list:
                key = (item.table, item.column)
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


# TODO
class TableSelector(DBInfoSelector):
    pass


# TODO
class ColumnSelector(DBInfoSelector):
    pass


class ColumnSelection(BaseModel):
    table: str = Field(description="表名")
    column: str = Field(description="字段名")


class SchemaSelection(BaseModel):
    selected_info: List[ColumnSelection] = Field(description="筛选出的表名和字段名，通常包含多个")
