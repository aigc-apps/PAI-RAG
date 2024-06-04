"""Tabular parser-Excel parser.

Contains parsers for tabular data files.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem

import pandas as pd
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PaiPandasExcelReader(BaseReader):
    r"""Pandas-based Excel parser.


    Args:
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.

        row_joiner (str): Separator to use for joining each row.
            Only used when `concat_rows=True`.
            Set to "\n" by default.

        pandas_config (dict): Options for the `pandas.read_excel` function call.
            Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
            for more information.
            Set to empty dict by default, this means pandas will try to figure
            out the separators, table head, etc. on its own.

    """

    def __init__(
        self,
        *args: Any,
        concat_rows: bool = True,
        row_joiner: str = "\n",
        pandas_config: dict = {},
        **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse Excel file. only process the first sheet"""
        if fs:
            with fs.open(file) as f:
                df = pd.read_excel(f, sheet_name=0, **self._pandas_config)
        else:
            df = pd.read_excel(file, sheet_name=0, **self._pandas_config)

        text_list = df.apply(
            lambda row: str(dict(zip(df.columns, row.astype(str)))), axis=1
        ).tolist()

        if self._concat_rows:
            return [
                Document(
                    text=(self._row_joiner).join(text_list), metadata=extra_info or {}
                )
            ]
        else:
            return [
                Document(text=text, metadata=extra_info or {}) for text in text_list
            ]
