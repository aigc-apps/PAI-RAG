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
import logging

logger = logging.getLogger(__name__)


class PaiCSVReader(BaseReader):
    """CSV parser.

    Args:
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.
        header （object）: None or int, list of int, default 0.
            Row (0-indexed) to use for the column labels of the parsed DataFrame. If a list of integers is passed those row
            positions will be combined into a MultiIndex. Use None if there is no header.

    """

    def __init__(
        self, *args: Any, concat_rows: bool = True, header: object = 0, **kwargs: Any
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._header = header

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse csv file.

        Returns:
            Union[str, List[str]]: a string or a List of strings.

        """
        try:
            import csv
        except ImportError:
            raise ImportError("csv module is required to read CSV files.")
        text_list = []
        headers = []
        data_lines = []
        data_line_start_index = 1
        if isinstance(self._header, list):
            data_line_start_index = max(self._header) + 1
        elif isinstance(self._header, int):
            data_line_start_index = self._header + 1
            self._header = [self._header]

        with open(file) as fp:
            csv_reader = csv.reader(fp)

            if self._header is None:
                for row in csv_reader:
                    text_list.append(", ".join(row))
            else:
                for i, row in enumerate(csv_reader):
                    if i in self._header:
                        headers.append(row)
                    elif i >= data_line_start_index:
                        data_lines.append(row)
                headers = [tuple(group) for group in zip(*headers)]
                for line in data_lines:
                    if len(line) == len(headers):
                        data_entry = str(dict(zip(headers, line)))
                        text_list.append(data_entry)

        metadata = {"filename": file.name, "extension": file.suffix}
        if extra_info:
            metadata = {**metadata, **extra_info}

        if self._concat_rows:
            return [Document(text="\n".join(text_list), metadata=metadata)]
        else:
            docs = []
            for i, text in enumerate(text_list):
                metadata["row_number"] = i + 1
                docs.append(Document(text=text, metadata=metadata))
            return docs


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

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse csv file."""
        if fs:
            with fs.open(file) as f:
                encoding = chardet.detect(f.read(100000))["encoding"]
                f.seek(0)
                if encoding is not None and "GB" in encoding.upper():
                    self._pandas_config["encoding"] = "GB18030"
                try:
                    df = pd.read_csv(f, **self._pandas_config)
                except UnicodeDecodeError:
                    logger.info(
                        f"Error: The file {file} encoding could not be decoded."
                    )
                    raise

        else:
            with open(file, "rb") as f:
                encoding = chardet.detect(f.read(100000))["encoding"]
                f.seek(0)
                if encoding is not None and "GB" in encoding.upper():
                    self._pandas_config["encoding"] = "GB18030"
                try:
                    df = pd.read_csv(file, **self._pandas_config)
                except UnicodeDecodeError:
                    logger.info(
                        f"Error: The file {file} encoding could not be decoded."
                    )
                    raise

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

        if self._concat_rows:
            return [
                Document(
                    text=(self._row_joiner).join(text_list), metadata=extra_info or {}
                )
            ]
        else:
            docs = []
            for i, text in enumerate(text_list):
                extra_info["row_number"] = i + 1
                docs.append(Document(text=text, metadata=extra_info))
            return docs
