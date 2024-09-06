import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate, PromptType
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType, TextNode
from llama_index.core.settings import Settings
from llama_index.core.callbacks.base import CallbackManager
from llama_index.experimental.query_engine.pandas.output_parser import (
    PandasInstructionParser,
)


logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION_STR = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

DEFAULT_PANDAS_TMPL = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

DEFAULT_PANDAS_PROMPT = PromptTemplate(
    DEFAULT_PANDAS_TMPL, prompt_type=PromptType.PANDAS
)


class PandasQueryRetriever(BaseRetriever):
    """
    Pandas query retriever

    Convert natural language to Pandas python code.

    Args:
        df (pd.DataFrame): Pandas dataframe to use
        instruction_str (Optional[str]): Instruction string to use
        output_processor (Optional[Callable[[str], str]]): Output processor
            A callable that takes in the output string, pandas DataFrame,
            and any output kwargs and returns a string.
            eg.kwargs["max_colwidth"] = [int] is used to set the length of text
            that each column can display during str(df). Set it to a higher number
            if there is possibly long text in the dataframe.
        pandas_prompt (Optional[BasePromptTemplate]): Pandas prompt to use.
        head (int): Number of rows to show in the table context.
        llm (Optional[LLM]): Language model to use.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        instruction_str: Optional[str] = None,
        instruction_parser: Optional[PandasInstructionParser] = None,
        pandas_prompt: Optional[BasePromptTemplate] = None,
        output_kwargs: Optional[dict] = None,
        head: int = 5,
        llm: Optional[LLM] = None,
        callback_manager: CallbackManager | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""

        self._df = df
        self._head = head
        self._pandas_prompt = pandas_prompt or DEFAULT_PANDAS_PROMPT
        self._instruction_str = instruction_str or DEFAULT_INSTRUCTION_STR
        self._instruction_parser = instruction_parser or PandasInstructionParser(
            self._df, output_kwargs or {}
        )
        self._llm = llm or Settings.llm

        super().__init__(callback_manager)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "pandas_prompt": self._pandas_prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "pandas_prompt" in prompts:
            self._pandas_prompt = prompts["pandas_prompt"]

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _get_table_context(self) -> str:
        """Get table context."""
        try:
            res = str(self._df.head(self._head))
        except Exception as e:
            logger.info(f"No dataframe provided, {e}")
            res = None
        return res

    def _retrieve(self, query_bundle: QueryType) -> List[NodeWithScore]:
        """Retrieve pandas instruction and pandas output."""
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        else:
            query_bundle = query_bundle

        context = self._get_table_context()
        logger.info(f"> Table head: {context}\n")

        # get executable python code
        pandas_response_str = self._llm.predict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_bundle.query_str,
            instruction_str=self._instruction_str,
        )
        logger.info(
            (
                f"> Pandas instructions (query code):\n"
                f"```\n{pandas_response_str}\n```\n"
            )
        )

        # get pandas output
        pandas_output = self._instruction_parser.parse(pandas_response_str)
        logger.info(f"> Pandas output: {pandas_output}\n")

        # check pandas output
        if (
            "There was an error running the output as Python code" in pandas_output
        ) or (pandas_output == "None"):
            pandas_output = str(self._df)

        retrieved_nodes = [
            NodeWithScore(
                node=TextNode(
                    text=str(pandas_output),
                    metadata={
                        "query_code_instruction": pandas_response_str,
                        "query_output": pandas_output,
                    },
                    excluded_embed_metadata_keys=[
                        "query_code_instruction",
                        "query_output",
                    ],
                    excluded_llm_metadata_keys=[
                        "query_code_instruction",
                        "query_output",
                    ],
                ),
                score=1.0,
            )
        ]
        return retrieved_nodes

    async def _aretrieve(self, query_bundle: QueryType) -> List[NodeWithScore]:
        """Async pandas instruction and pandas output."""
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_bundle)
        else:
            query_bundle = query_bundle

        context = self._get_table_context()
        logger.info(f"> Async Table head: {context}\n")

        # get executable python code
        pandas_response_str = await self._llm.apredict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_bundle.query_str,
            instruction_str=self._instruction_str,
        )
        logger.info(
            (
                f"> Async Pandas instructions (query code):\n"
                f"```\n{pandas_response_str}\n```\n"
            )
        )

        # get pandas output
        pandas_output = self._instruction_parser.parse(pandas_response_str)
        logger.info(f"> Async Pandas output: {pandas_output}\n")

        # check pandas output
        if (
            "There was an error running the output as Python code" in pandas_output
        ) or (pandas_output == "None"):
            pandas_output = str(self._df)

        retrieved_nodes = [
            NodeWithScore(
                node=TextNode(
                    text=str(pandas_output),
                    metadata={
                        "query_code_instruction": pandas_response_str,
                        "query_output": pandas_output,
                    },
                    excluded_embed_metadata_keys=[
                        "query_code_instruction",
                        "query_output",
                    ],
                    excluded_llm_metadata_keys=[
                        "query_code_instruction",
                        "query_output",
                    ],
                ),
                score=1.0,
            )
        ]
        return retrieved_nodes
