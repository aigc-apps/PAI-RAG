import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock
from openpyxl import Workbook
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.file_task import KbFileTaskEntity


def _make_xlsx_bytes(header, rows):
    """Create xlsx bytes from header and rows."""
    wb = Workbook()
    ws = wb.active
    ws.append(header)
    for row in rows:
        ws.append(row)
    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def _make_file_entity(**kwargs):
    defaults = dict(
        id="file1",
        kb_id="kb1",
        file_name="test.xlsx",
        file_path="kb1/docs/test.xlsx",
        file_version=1,
        tenant_id="t1",
    )
    defaults.update(kwargs)
    return KbFileEntity(**defaults)


class TestExcelSplit:
    @patch("rag.split.excel_split.file_store")
    def test_small_file_single_part(self, mock_file_store):
        xlsx_data = _make_xlsx_bytes(["col1", "col2"], [["a", "b"], ["c", "d"]])
        mock_file_store.read.return_value = xlsx_data

        from rag.split.excel_split import split_excel

        entity = _make_file_entity()
        tasks = list(split_excel(entity))
        assert len(tasks) == 1
        assert tasks[0].file_part == 0

    @patch("rag.split.excel_split.MAX_PART_ROW_NUM", 2)
    @patch("rag.split.excel_split.file_store")
    def test_large_file_multiple_parts(self, mock_file_store):
        rows = [["val1", "val2"] for _ in range(5)]
        xlsx_data = _make_xlsx_bytes(["col1", "col2"], rows)
        mock_file_store.read.return_value = xlsx_data

        upload_result = MagicMock()
        upload_result.file_path = "kb1/docs/test_Part0001.xlsx"
        mock_file_store.write.return_value = upload_result

        from rag.split.excel_split import split_excel

        entity = _make_file_entity()
        tasks = list(split_excel(entity))
        assert len(tasks) >= 2
        assert all(isinstance(t, KbFileTaskEntity) for t in tasks)

    @patch("rag.split.excel_split.file_store")
    def test_single_row_sheet(self, mock_file_store):
        """Sheet with only header row (no data rows) should yield single task."""
        wb = Workbook()
        ws = wb.active
        ws.append(["col1", "col2"])
        # max_row=1 means just the header
        buf = BytesIO()
        wb.save(buf)
        buf.seek(0)
        mock_file_store.read.return_value = buf

        from rag.split.excel_split import split_excel

        entity = _make_file_entity()
        tasks = list(split_excel(entity))
        assert len(tasks) == 1
        assert tasks[0].file_part == 0

    @patch("rag.split.excel_split.file_store")
    def test_header_only(self, mock_file_store):
        xlsx_data = _make_xlsx_bytes(["col1", "col2"], [])
        mock_file_store.read.return_value = xlsx_data

        from rag.split.excel_split import split_excel

        entity = _make_file_entity()
        tasks = list(split_excel(entity))
        assert len(tasks) == 1
        assert tasks[0].file_part == 0
