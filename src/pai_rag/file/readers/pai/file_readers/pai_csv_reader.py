"""Tabular parser-CSV parser.

Contains parsers for tabular data files.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem

import pandas as pd
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


import chardet
import os

from pai_rag.utils.nodeid_util import compute_node_id


class PaiPandasCSVReader(BaseReader):
    r"""Pandas-based CSV parser.

    Parses CSVs using the separator detection from Pandas `read_csv`function.
    If special parameters are required, use the `pandas_config` dict.

    Args:
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.

        row_joiner (str): Separator to use for joining each row.
            Only used when `concat_rows=True`.
            Set to "\n" by default.

        pandas_config (dict): Options for the `pandas.read_csv` function call.
            Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
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
        format_sheet_data_to_json: bool = False,
        sheet_column_filters: List[str] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config
        self._format_sheet_data_to_json = format_sheet_data_to_json
        self._sheet_column_filters = sheet_column_filters

    def _read_file(self, file: Any):
        encoding = chardet.detect(file.read(10000))["encoding"]
        file.seek(0)
        encoding = "utf-8"
        if encoding is not None and "GB" in encoding.upper():
            encoding = "GB18030"

        df = pd.read_csv(file, encoding=encoding, **self._pandas_config)
        return df

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse csv file."""
        if fs:
            with fs.open(file) as f:
                df = self._read_file(f)
        else:
            with open(file, "rb") as f:
                df = self._read_file(f)

        if self._sheet_column_filters:
            df = df[self._sheet_column_filters]

        if self._format_sheet_data_to_json:
            text_list = df.apply(
                lambda row: str(dict(zip(df.columns, row.astype(str)))), axis=1
            ).tolist()
        else:
            text_list = [
                "\n".join([f"{k}:{v}" for k, v in record.items()])
                for record in df.to_dict("records")
            ]

        file_name = os.path.basename(file)
        extra_info = extra_info or {}
        extra_info["file_path"] = str(file)
        extra_info["file_name"] = file_name

        if self._concat_rows:
            doc_id = compute_node_id(i=0, file_name=file_name)
            return [
                Document(
                    id_=doc_id,
                    text=(self._row_joiner).join(text_list),
                    metadata=extra_info,
                )
            ]
        else:
            docs = []
            for i, text in enumerate(text_list):
                doc_id = compute_node_id(i=i, file_name=file_name)
                extra_info["row_number"] = i + 1
                docs.append(Document(id_=doc_id, text=text, metadata=extra_info))
            return docs
