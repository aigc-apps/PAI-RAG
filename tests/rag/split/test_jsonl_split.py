import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import json
import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.file_task import KbFileTaskEntity


def _make_jsonl_bytes(records):
    """Create JSONL bytes from a list of dicts."""
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    return BytesIO("\n".join(lines).encode("utf-8"))


def _make_file_entity(**kwargs):
    defaults = dict(
        id="file1",
        kb_id="kb1",
        file_name="test.jsonl",
        file_path="kb1/docs/test.jsonl",
        file_version=1,
        tenant_id="t1",
    )
    defaults.update(kwargs)
    return KbFileEntity(**defaults)


class TestJsonlSplit:
    @patch("rag.split.jsonl_split.file_store")
    def test_small_file_single_part(self, mock_file_store):
        records = [{"key": f"val{i}"} for i in range(5)]
        mock_file_store.read.return_value = _make_jsonl_bytes(records)

        from rag.split.jsonl_split import split_jsonl

        entity = _make_file_entity()
        tasks = list(split_jsonl(entity))
        assert len(tasks) == 1
        assert tasks[0].file_part == 0
        assert tasks[0].file_path == entity.file_path

    @patch("rag.split.jsonl_split.MAX_PART_ROW_NUM", 2)
    @patch("rag.split.jsonl_split.file_store")
    def test_large_file_multiple_parts(self, mock_file_store):
        records = [{"key": f"val{i}"} for i in range(5)]
        mock_file_store.read.return_value = _make_jsonl_bytes(records)

        upload_result = MagicMock()
        upload_result.file_path = "kb1/docs/test_Part0001.jsonl"
        mock_file_store.write.return_value = upload_result

        from rag.split.jsonl_split import split_jsonl

        entity = _make_file_entity()
        tasks = list(split_jsonl(entity))
        assert len(tasks) >= 2
        assert all(isinstance(t, KbFileTaskEntity) for t in tasks)

    @patch("rag.split.jsonl_split.file_store")
    def test_empty_file(self, mock_file_store):
        mock_file_store.read.return_value = BytesIO(b"")

        from rag.split.jsonl_split import split_jsonl

        entity = _make_file_entity()
        tasks = list(split_jsonl(entity))
        # Empty file with no lines yields single task with part=0
        assert len(tasks) == 1
        assert tasks[0].file_part == 0

    @patch("rag.split.jsonl_split.file_store")
    def test_blank_lines_skipped(self, mock_file_store):
        content = b'{"a":1}\n\n\n{"b":2}\n'
        mock_file_store.read.return_value = BytesIO(content)

        from rag.split.jsonl_split import split_jsonl

        entity = _make_file_entity()
        tasks = list(split_jsonl(entity))
        assert len(tasks) == 1
        assert tasks[0].file_part == 0
