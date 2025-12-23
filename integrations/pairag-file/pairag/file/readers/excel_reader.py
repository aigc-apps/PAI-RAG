"""Tabular parser-Excel parser.

Contains parsers for tabular data files.

"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem
from loguru import logger
from openpyxl import load_workbook

import pandas as pd
from llama_index.core.schema import Document
from pairag.file.readers.base import BaseReader
from pairag.file.models.file_item import FileItem


class ExcelReader(BaseReader):

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
        self._row_joiner = row_joiner  if row_joiner is not None else "\n"
        self._header_max = header_max if header_max is not None else 0
        self._format_sheet_data_to_json = format_sheet_data_to_json if format_sheet_data_to_json is not None else False
        self._sheet_column_filters = sheet_column_filters if sheet_column_filters is not None else None
        self._pandas_config = {'header': header_max} if header_max is not None else {}

    def read_xlsx(
        self,
        file: Path,
        fs: Optional[AbstractFileSystem] = None,
    ):
        """Parse Excel file。"""
        if fs:
            with fs.open(file) as f:
                excel = pd.ExcelFile(
                    load_workbook(f, data_only=True), engine="openpyxl"
                )
        else:
            excel = pd.ExcelFile(load_workbook(file, data_only=True), engine="openpyxl")
        sheet_name = excel.sheet_names[0]
        sheet = excel.book[sheet_name]
        df = excel.parse(sheet_name, **self._pandas_config)


        for item in sheet.merged_cells:
            top_col, top_row, bottom_col, bottom_row = item.bounds
            base_value = item.start_cell.value
            # Convert 1-based index to 0-based index
            top_row -= 1
            top_col -= 1
            # Since the previous lines are set as headers, the coordinates need to be adjusted here.
            if self._header_max is not None and self._header_max > 0:
                top_row -= self._header_max + 1
                bottom_row -= self._header_max + 1

            df.iloc[top_row:bottom_row, top_col:bottom_col] = base_value
        return df

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse Excel file. only process the first sheet"""

        logger.info(f"Parsing workbook {file}.")
        
        # Convert .xls to .xlsx if needed
        file_path = Path(file)
        if file_path.suffix.lower() == ".xls":
            tmp_file_dir = Path("/tmp/pairag_excels")
            tmp_file_dir.mkdir(parents=True, exist_ok=True)
            workbook_file = tmp_file_dir / f"{file_path.stem}.xlsx"
            logger.info(f"Transfer {file} to {workbook_file}.")
            pd.read_excel(file, engine="xlrd").to_excel(workbook_file, index=False, engine="openpyxl")
        else:
            workbook_file = file

        df = self.read_xlsx(workbook_file, fs)

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
            logger.info(f"Parsed workbook {workbook_file} into single document.")

            return [
                Document(
                    text=(self._row_joiner).join(text_list), metadata=extra_info or {}
                )
            ]
        else:
            docs = []
            extra_info = extra_info or {}
            for i, text in enumerate(text_list):
                row_metadata = extra_info.copy()
                row_metadata["row_number"] = i + 1
                docs.append(Document(text=text, metadata=row_metadata))

            logger.info(f"Parsed workbook {workbook_file} into {len(docs)} documents.")
            return docs

    def read(self, file_item: FileItem) -> List[Document]:
        """Read Excel file from FileItem."""
        file_path = Path(file_item.file_path)
        extra_info = file_item.metadata()
        return self.load_data(file_path, extra_info=extra_info)