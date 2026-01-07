"""Tabular parser-Excel parser for FAQ file。

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


class FAQReader(BaseReader):

    def __init__(
        self,
        *args: Any,
        header_index_max: Optional[int] = 0,
        question_column_index: Optional[int] = 0,
        answer_column_index: Optional[int] = 1,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._question_column_index = question_column_index if question_column_index is not None else 0
        self._answer_column_index = answer_column_index if answer_column_index is not None else 1
        self._header_index_max = header_index_max  # Allow None to indicate no header row
        # When header_index_max is None, pandas will use numeric column indices (0, 1, 2, ...)
        self._pandas_config = {'header': None} if self._header_index_max is None else {'header': self._header_index_max}

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
            if self._header_index_max is not None and self._header_index_max > 0:
                top_row -= self._header_index_max + 1
                bottom_row -= self._header_index_max + 1

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
        return self._process_dataframe(df, extra_info, str(workbook_file))
    
    def _process_dataframe(self, df: pd.DataFrame, extra_info: Optional[Dict] = None, file_name: Optional[str] = None) -> List[Document]:
        """Process DataFrame and create FAQ documents."""

        # Get question and answer columns by index
        if len(df.columns) <= self._question_column_index:
            raise ValueError(f"Question column index {self._question_column_index} is out of range. DataFrame has {len(df.columns)} columns.")
        if len(df.columns) <= self._answer_column_index:
            raise ValueError(f"Answer column index {self._answer_column_index} is out of range. DataFrame has {len(df.columns)} columns.")
        
        question_column = df.columns[self._question_column_index]
        answer_column = df.columns[self._answer_column_index]

        # Build documents for each row
        docs = []
        extra_info = extra_info or {}
        
        for i, row in df.iterrows():
            question = str(row[question_column]) if pd.notna(row[question_column]) else ""
            answer = str(row[answer_column]) if pd.notna(row[answer_column]) else ""
            
            if not question.strip() and not answer.strip():
                continue
            
            
            
            chunk_text = f"问题: {question}\n答案: {answer}"
            
            row_metadata = extra_info.copy()
            row_metadata["row_number"] = i + 1
            row_metadata["question"] = question
            row_metadata["answer"] = answer
            
            docs.append(Document(text=chunk_text, metadata=row_metadata))

        file_display_name = file_name if file_name else "file"
        logger.info(f"Parsed workbook {file_display_name} into {len(docs)} FAQ documents.")
        return docs

    def read(self, file_item: FileItem) -> List[Document]:
        """Read Excel file from FileItem."""
        extra_info = file_item.metadata()
        
        file_path = Path(file_item.file_path)
        return self.load_data(file_path, extra_info=extra_info)
        