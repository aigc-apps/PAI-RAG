"""Tabular parser-CSV parser.

Contains parsers for tabular data files.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem

import pandas as pd
from llama_index.core.schema import Document
from pairag.file.readers.base import BaseReader
from pairag.file.models.file_item import FileItem
import chardet
import os


class CSVReader(BaseReader):

    def __init__(
        self,
        *args: Any,
        concat_rows: Optional[bool] = False,
        row_joiner: Optional[str] = "\n",
        header_max: Optional[int] = 0,
        format_sheet_data_to_json: Optional[bool] = False,
        sheet_column_filters: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows if concat_rows is not None else False
        self._row_joiner = row_joiner if row_joiner is not None else "\n"
        self._pandas_config = {'header': header_max} if header_max is not None else {}
        self._format_sheet_data_to_json = format_sheet_data_to_json if format_sheet_data_to_json is not None else False
        self._sheet_column_filters = sheet_column_filters if sheet_column_filters is not None else None

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
            return [
                Document(
                    text=(self._row_joiner).join(text_list),
                    metadata=extra_info,
                )
            ]
        else:
            docs = []
            extra_info = extra_info or {}
            for i, text in enumerate(text_list):
                row_metadata = extra_info.copy()
                row_metadata["row_number"] = i + 1
                docs.append(Document(text=text, metadata=row_metadata))
            return docs

    def read(self, file_item: FileItem) -> List[Document]:
        """Read CSV file from FileItem."""
        file_path = Path(file_item.file_path)
        extra_info = file_item.metadata()
        return self.load_data(file_path, extra_info=extra_info)