"""Tabular parser-CSV parser.


Contains parsers for tabular data files.

"""

from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional
from loguru import logger

import pandas as pd
from llama_index.core.schema import Document
import os
from pairag.file.readers.base import BaseReader, FileItem, Document, List
import charset_normalizer
from pairag.file.utils.text_utils import replace_consecutive_spaces
class CSVReader(BaseReader):


    def __init__(
        self,
        *args: Any,
        concat_rows: Optional[bool] = False,
        row_joiner: Optional[str] = "\n",
        header_index_max: Optional[int] = 0,
        format_sheet_data_to_json: Optional[bool] = False,
        sheet_column_filters: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows if concat_rows is not None else False
        self._row_joiner = row_joiner if row_joiner is not None else "\n"
        self._header_index_max = header_index_max if header_index_max is not None else 0
        self._format_sheet_data_to_json = format_sheet_data_to_json if format_sheet_data_to_json is not None else False
        self._sheet_column_filters = sheet_column_filters if sheet_column_filters is not None else None
        self._pandas_config = {'header': self._header_index_max} if self._header_index_max is not None else {}

    def _read_file(self, file: BinaryIO):
        """Read CSV file from binary file object."""
        encoding = charset_normalizer.detect(file.read(10240)).get("encoding")
        file.seek(0)
        if not encoding:
            logger.error("Failed to detect encoding, using utf-8")
            encoding = "utf-8"
        
        df = pd.read_csv(file, encoding=encoding, **self._pandas_config)
        return df

    def read(self, file_item: FileItem) -> List[Document]:
        """Read CSV file from FileItem."""
        extra_info = file_item.metadata()
        
        # Use file_item.file directly, similar to Csv2MdReader
        file_item.file.seek(0)
        df = self._read_file(file_item.file)
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
        
        for text in text_list:
            text = replace_consecutive_spaces(text)

        extra_info = extra_info or {}
        extra_info["file_path"] = file_item.file_path
        extra_info["file_name"] = file_item.file_name

        if self._concat_rows:
            return [
                Document(
                    id_=file_item.id,
                    text=(self._row_joiner).join(text_list),
                    metadata=extra_info,
                )
            ]
        else:
            docs = []
            for i, text in enumerate(text_list):
                row_metadata = extra_info.copy()
                row_metadata["row_number"] = i + 1
                docs.append(Document(id_=file_item.id, text=text, metadata=row_metadata))
            return docs