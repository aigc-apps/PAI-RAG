import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from utils.upload_file_utils import load_eval_dataset_from_local_path


class TestLoadEvalDatasetFromLocalPath:
    def test_valid_jsonl_file(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"input": "q1", "output": "a1"}) + "\n"
            + json.dumps({"input": "q2", "output": "a2"}) + "\n"
        )
        result = load_eval_dataset_from_local_path(str(f))
        assert len(result) == 2
        assert result[0]["input"] == "q1"

    def test_skips_lines_without_input(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"input": "q1"}) + "\n"
            + json.dumps({"other": "val"}) + "\n"
        )
        result = load_eval_dataset_from_local_path(str(f))
        assert len(result) == 1

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"input": "q1"}) + "\n\n\n"
            + json.dumps({"input": "q2"}) + "\n"
        )
        result = load_eval_dataset_from_local_path(str(f))
        assert len(result) == 2

    def test_skips_invalid_json_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"input": "q1"}) + "\n"
            + "not valid json\n"
        )
        result = load_eval_dataset_from_local_path(str(f))
        assert len(result) == 1

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_eval_dataset_from_local_path("/nonexistent/path.jsonl")

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        result = load_eval_dataset_from_local_path(str(f))
        assert result == []
