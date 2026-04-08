import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import csv
import pytest
from io import BytesIO, TextIOWrapper
from unittest.mock import patch, MagicMock
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.file_task import KbFileTaskEntity


def _make_csv_bytes(header, rows):
    """Helper to create CSV bytes from header and rows."""
    buf = BytesIO()
    tw = TextIOWrapper(buf, encoding="utf-8", newline="")
    writer = csv.writer(tw)
    writer.writerow(header)
    writer.writerows(rows)
    tw.flush()
    buf = tw.detach()
    buf.seek(0)
    return buf


def _make_file_entity(**kwargs):
    defaults = dict(
        id="file1",
        kb_id="kb1",
        file_name="test.csv",
        file_path="kb1/docs/test.csv",
        file_version=1,
        tenant_id="t1",
    )
    defaults.update(kwargs)
    return KbFileEntity(**defaults)


class TestCsvSplit:
    @patch("rag.split.csv_split.file_store")
    def test_small_file_single_part(self, mock_file_store):
        """File with rows < MAX_PART_ROW_NUM should yield single task with file_part=0."""
        csv_data = _make_csv_bytes(["col1", "col2"], [["a", "b"], ["c", "d"]])
        mock_file_store.read.return_value = csv_data

        from rag.split.csv_split import split_csv

        entity = _make_file_entity()
        tasks = list(split_csv(entity))
        assert len(tasks) == 1
        assert tasks[0].file_part == 0
        assert tasks[0].file_path == entity.file_path

    @patch("rag.split.csv_split.file_store")
    def test_empty_csv_no_tasks(self, mock_file_store):
        """Empty CSV (no header) should yield no tasks."""
        csv_data = BytesIO(b"")
        mock_file_store.read.return_value = csv_data

        from rag.split.csv_split import split_csv

        entity = _make_file_entity()
        tasks = list(split_csv(entity))
        assert len(tasks) == 0

    @patch("rag.split.csv_split.MAX_PART_ROW_NUM", 2)
    @patch("rag.split.csv_split.file_store")
    def test_large_file_multiple_parts(self, mock_file_store):
        """File with rows > MAX_PART_ROW_NUM should be split."""
        rows = [["val1", "val2"] for _ in range(5)]
        csv_data = _make_csv_bytes(["col1", "col2"], rows)
        mock_file_store.read.return_value = csv_data

        upload_result = MagicMock()
        upload_result.file_path = "kb1/docs/test_Part0001.csv"
        mock_file_store.write.return_value = upload_result

        from rag.split.csv_split import split_csv

        entity = _make_file_entity()
        tasks = list(split_csv(entity))
        assert len(tasks) >= 2
        assert all(isinstance(t, KbFileTaskEntity) for t in tasks)

    @patch("rag.split.csv_split.file_store")
    def test_header_only_csv(self, mock_file_store):
        """CSV with only header row should yield single task with file_part=0."""
        csv_data = _make_csv_bytes(["col1", "col2"], [])
        mock_file_store.read.return_value = csv_data

        from rag.split.csv_split import split_csv

        entity = _make_file_entity()
        tasks = list(split_csv(entity))
        assert len(tasks) == 1
        assert tasks[0].file_part == 0
