import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from pydantic import ValidationError
from common.chat.models import (
    Condition,
    MetadataFilteringCondition,
    RetrievalSetting,
    RetrievalRequest,
)


class TestCondition:
    def test_in_operator_with_list(self):
        c = Condition(name="category", comparison_operator="in", value=["a", "b"])
        assert c.value == ["a", "b"]

    def test_in_operator_with_str_raises(self):
        with pytest.raises(ValidationError):
            Condition(name="category", comparison_operator="in", value="a")

    def test_not_in_operator_with_list(self):
        c = Condition(name="category", comparison_operator="not in", value=["x"])
        assert c.value == ["x"]

    def test_is_operator_with_str(self):
        c = Condition(name="status", comparison_operator="is", value="active")
        assert c.value == "active"


class TestRetrievalSetting:
    def test_score_threshold_backfills_similarity(self):
        s = RetrievalSetting(score_threshold=0.5)
        assert s.similarity_threshold == 0.5

    def test_similarity_threshold_preferred(self):
        s = RetrievalSetting(similarity_threshold=0.8, score_threshold=0.5)
        assert s.similarity_threshold == 0.8


class TestMetadataFilteringCondition:
    def test_nested_condition_groups(self):
        group = MetadataFilteringCondition(
            logical_operator="or",
            condition_groups=[
                MetadataFilteringCondition(
                    conditions=[
                        Condition(name="cat", comparison_operator="is", value="A")
                    ]
                ),
                MetadataFilteringCondition(
                    conditions=[
                        Condition(name="cat", comparison_operator="is", value="B")
                    ]
                ),
            ],
        )
        assert len(group.condition_groups) == 2
        assert group.logical_operator == "or"


class TestRetrievalRequest:
    def test_basic_construction(self):
        req = RetrievalRequest(query="test query", knowledge_id="kb1")
        assert req.query == "test query"
        assert req.knowledge_id == "kb1"
