"""Unit tests for metadata filter — especially nested condition_groups support."""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from common.chat.models import Condition, MetadataFilteringCondition
from rag.metadata_filter import (
    _build_metadata_condition_,
    _build_metadata_filter_recursive,
)
from sqlalchemy.sql.elements import BooleanClauseList, BinaryExpression


# ───────────────────────────────────────────────────
# Model serialization / deserialization tests
# ───────────────────────────────────────────────────


class TestMetadataFilteringConditionModel:
    """Test Pydantic model for MetadataFilteringCondition with nested support."""

    def test_flat_conditions_backward_compat(self):
        """Old-style flat conditions should still parse correctly."""
        raw = {
            "logical_operator": "and",
            "conditions": [
                {"name": "department", "comparison_operator": "is", "value": "it"}
            ],
        }
        model = MetadataFilteringCondition.model_validate(raw)
        assert model.logical_operator == "and"
        assert len(model.conditions) == 1
        assert model.conditions[0].name == "department"
        assert model.condition_groups is None

    def test_nested_condition_groups(self):
        """(category='COMMON' OR category='PC') AND language='en-US'"""
        raw = {
            "logical_operator": "and",
            "condition_groups": [
                {
                    "logical_operator": "or",
                    "conditions": [
                        {"name": "category", "comparison_operator": "is", "value": "COMMON"},
                        {"name": "category", "comparison_operator": "is", "value": "PC"},
                    ],
                },
                {
                    "conditions": [
                        {"name": "language", "comparison_operator": "is", "value": "en-US"},
                    ],
                },
            ],
        }
        model = MetadataFilteringCondition.model_validate(raw)
        assert model.condition_groups is not None
        assert len(model.condition_groups) == 2
        assert model.condition_groups[0].logical_operator == "or"
        assert len(model.condition_groups[0].conditions) == 2
        assert model.condition_groups[1].conditions[0].value == "en-US"

    def test_mixed_conditions_and_groups(self):
        """conditions and condition_groups at same level."""
        raw = {
            "logical_operator": "and",
            "conditions": [
                {"name": "status", "comparison_operator": "is", "value": "active"},
            ],
            "condition_groups": [
                {
                    "logical_operator": "or",
                    "conditions": [
                        {"name": "category", "comparison_operator": "is", "value": "A"},
                        {"name": "category", "comparison_operator": "is", "value": "B"},
                    ],
                },
            ],
        }
        model = MetadataFilteringCondition.model_validate(raw)
        assert len(model.conditions) == 1
        assert len(model.condition_groups) == 1

    def test_deeply_nested(self):
        """3-level nesting should work."""
        raw = {
            "logical_operator": "and",
            "condition_groups": [
                {
                    "logical_operator": "or",
                    "condition_groups": [
                        {
                            "logical_operator": "and",
                            "conditions": [
                                {"name": "a", "comparison_operator": "is", "value": "1"},
                                {"name": "b", "comparison_operator": ">", "value": 10},
                            ],
                        },
                        {
                            "conditions": [
                                {"name": "c", "comparison_operator": "is", "value": "x"},
                            ],
                        },
                    ],
                },
            ],
        }
        model = MetadataFilteringCondition.model_validate(raw)
        inner = model.condition_groups[0].condition_groups[0]
        assert inner.logical_operator == "and"
        assert len(inner.conditions) == 2

    def test_roundtrip_json(self):
        """model_dump → model_validate round-trip preserves structure."""
        original = MetadataFilteringCondition(
            logical_operator="and",
            condition_groups=[
                MetadataFilteringCondition(
                    logical_operator="or",
                    conditions=[
                        Condition(name="x", comparison_operator="is", value="1"),
                        Condition(name="x", comparison_operator="is", value="2"),
                    ],
                ),
            ],
        )
        raw = json.loads(original.model_dump_json())
        restored = MetadataFilteringCondition.model_validate(raw)
        assert restored.condition_groups[0].logical_operator == "or"
        assert len(restored.condition_groups[0].conditions) == 2

    def test_empty_conditions_and_groups(self):
        """Both fields empty/None."""
        model = MetadataFilteringCondition()
        assert model.conditions is None
        assert model.condition_groups is None

    def test_default_logical_operator_is_and(self):
        model = MetadataFilteringCondition(
            conditions=[Condition(name="a", comparison_operator="is", value="1")]
        )
        assert model.logical_operator == "and"


# ───────────────────────────────────────────────────
# _build_metadata_condition_ (leaf level) tests
# ───────────────────────────────────────────────────


class TestBuildMetadataCondition:
    """Test individual condition → SQLAlchemy clause conversion."""

    def test_is_string(self):
        cond = Condition(name="dept", comparison_operator="is", value="it")
        result = _build_metadata_condition_(cond)
        assert result is not None
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "dept" in compiled

    def test_equals_number(self):
        cond = Condition(name="score", comparison_operator="=", value=90)
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_contains(self):
        cond = Condition(name="tags", comparison_operator="contains", value="python")
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_empty_operator(self):
        cond = Condition(name="field", comparison_operator="empty")
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_not_empty_operator(self):
        cond = Condition(name="field", comparison_operator="not empty")
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_null_value_non_empty_operator_returns_none(self):
        cond = Condition(name="field", comparison_operator="is", value=None)
        result = _build_metadata_condition_(cond)
        assert result is None

    def test_empty_string_value_returns_none(self):
        cond = Condition(name="field", comparison_operator="contains", value="")
        result = _build_metadata_condition_(cond)
        assert result is None

    def test_unknown_operator_returns_none(self):
        cond = Condition(name="field", comparison_operator="is", value="x")
        # Monkey-patch to simulate unknown operator
        cond.comparison_operator = "unknown_op"
        result = _build_metadata_condition_(cond)
        assert result is None

    def test_greater_than(self):
        cond = Condition(name="price", comparison_operator=">", value=100)
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_less_than_or_equal(self):
        cond = Condition(name="price", comparison_operator="≤", value=50)
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_start_with(self):
        cond = Condition(name="name", comparison_operator="start with", value="doc")
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_end_with(self):
        cond = Condition(name="name", comparison_operator="end with", value=".pdf")
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_is_not_string(self):
        cond = Condition(name="dept", comparison_operator="is not", value="hr")
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_not_equal_number(self):
        cond = Condition(name="score", comparison_operator="≠", value=0)
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_before(self):
        cond = Condition(name="created", comparison_operator="before", value=1700000000)
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_after(self):
        cond = Condition(name="created", comparison_operator="after", value=1600000000)
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_in_operator(self):
        cond = Condition(name="category", comparison_operator="in", value=["A", "B", "C"])
        result = _build_metadata_condition_(cond)
        assert result is not None
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "IN" in compiled

    def test_not_in_operator(self):
        cond = Condition(name="category", comparison_operator="not in", value=["X", "Y"])
        result = _build_metadata_condition_(cond)
        assert result is not None

    def test_in_operator_non_list_raises_validation_error(self):
        """in operator with non-list value should be rejected at model level."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="requires a list"):
            Condition(name="category", comparison_operator="in", value="single_value")

    def test_not_in_operator_non_list_raises_validation_error(self):
        """not in operator with non-list value should be rejected at model level."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="requires a list"):
            Condition(name="category", comparison_operator="not in", value="single_value")


# ───────────────────────────────────────────────────
# _build_metadata_filter_recursive tests
# ───────────────────────────────────────────────────


class TestBuildMetadataFilterRecursive:
    """Test recursive filter builder produces correct SQLAlchemy clause trees."""

    def test_flat_and(self):
        """Flat AND of two conditions."""
        mf = MetadataFilteringCondition(
            logical_operator="and",
            conditions=[
                Condition(name="a", comparison_operator="is", value="1"),
                Condition(name="b", comparison_operator="is", value="2"),
            ],
        )
        result = _build_metadata_filter_recursive(mf)
        assert result is not None
        # AND of 2 → BooleanClauseList
        assert isinstance(result, BooleanClauseList)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "AND" in compiled

    def test_flat_or(self):
        """Flat OR of two conditions."""
        mf = MetadataFilteringCondition(
            logical_operator="or",
            conditions=[
                Condition(name="a", comparison_operator="is", value="1"),
                Condition(name="b", comparison_operator="is", value="2"),
            ],
        )
        result = _build_metadata_filter_recursive(mf)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "OR" in compiled

    def test_single_condition_no_wrapping(self):
        """Single condition should be returned as-is, not wrapped in AND/OR."""
        mf = MetadataFilteringCondition(
            logical_operator="and",
            conditions=[
                Condition(name="a", comparison_operator="is", value="1"),
            ],
        )
        result = _build_metadata_filter_recursive(mf)
        assert result is not None
        # Single condition → BinaryExpression, not BooleanClauseList
        assert not isinstance(result, BooleanClauseList)

    def test_nested_or_inside_and(self):
        """(category='COMMON' OR category='PC') AND language='en-US'"""
        mf = MetadataFilteringCondition(
            logical_operator="and",
            condition_groups=[
                MetadataFilteringCondition(
                    logical_operator="or",
                    conditions=[
                        Condition(name="category", comparison_operator="is", value="COMMON"),
                        Condition(name="category", comparison_operator="is", value="PC"),
                    ],
                ),
                MetadataFilteringCondition(
                    conditions=[
                        Condition(name="language", comparison_operator="is", value="en-US"),
                    ],
                ),
            ],
        )
        result = _build_metadata_filter_recursive(mf)
        assert result is not None
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        # Should have both AND and OR
        assert "AND" in compiled
        assert "OR" in compiled

    def test_mixed_conditions_and_groups(self):
        """status='active' AND (cat='A' OR cat='B')"""
        mf = MetadataFilteringCondition(
            logical_operator="and",
            conditions=[
                Condition(name="status", comparison_operator="is", value="active"),
            ],
            condition_groups=[
                MetadataFilteringCondition(
                    logical_operator="or",
                    conditions=[
                        Condition(name="cat", comparison_operator="is", value="A"),
                        Condition(name="cat", comparison_operator="is", value="B"),
                    ],
                ),
            ],
        )
        result = _build_metadata_filter_recursive(mf)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "AND" in compiled
        assert "OR" in compiled

    def test_empty_filter_returns_none(self):
        mf = MetadataFilteringCondition()
        result = _build_metadata_filter_recursive(mf)
        assert result is None

    def test_all_conditions_invalid_returns_none(self):
        """All conditions have null values (non-empty operator) → all skipped."""
        mf = MetadataFilteringCondition(
            conditions=[
                Condition(name="a", comparison_operator="is", value=None),
                Condition(name="b", comparison_operator="contains", value=""),
            ],
        )
        result = _build_metadata_filter_recursive(mf)
        assert result is None

    def test_three_level_nesting(self):
        """(a=1 AND b>10) OR c='x', wrapped in top-level AND with d='y'"""
        mf = MetadataFilteringCondition(
            logical_operator="and",
            conditions=[
                Condition(name="d", comparison_operator="is", value="y"),
            ],
            condition_groups=[
                MetadataFilteringCondition(
                    logical_operator="or",
                    condition_groups=[
                        MetadataFilteringCondition(
                            logical_operator="and",
                            conditions=[
                                Condition(name="a", comparison_operator="=", value=1),
                                Condition(name="b", comparison_operator=">", value=10),
                            ],
                        ),
                        MetadataFilteringCondition(
                            conditions=[
                                Condition(name="c", comparison_operator="is", value="x"),
                            ],
                        ),
                    ],
                ),
            ],
        )
        result = _build_metadata_filter_recursive(mf)
        assert result is not None
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "AND" in compiled
        assert "OR" in compiled

    def test_nested_with_empty_group_skipped(self):
        """Empty nested group should be skipped, not break anything."""
        mf = MetadataFilteringCondition(
            logical_operator="and",
            conditions=[
                Condition(name="a", comparison_operator="is", value="1"),
            ],
            condition_groups=[
                MetadataFilteringCondition(),  # empty group
            ],
        )
        result = _build_metadata_filter_recursive(mf)
        assert result is not None
        # Should be just the single condition, empty group skipped
        assert not isinstance(result, BooleanClauseList)

    def test_only_condition_groups_no_conditions_not_empty(self):
        """Filter with only condition_groups (no conditions) should NOT be treated as empty."""
        mf = MetadataFilteringCondition(
            logical_operator="and",
            condition_groups=[
                MetadataFilteringCondition(
                    conditions=[
                        Condition(name="a", comparison_operator="is", value="1"),
                    ],
                ),
            ],
        )
        result = _build_metadata_filter_recursive(mf)
        assert result is not None

    def test_exceeds_max_depth_raises_error(self):
        """Nesting beyond max_depth should raise ValueError."""
        # Build a chain of 6 levels deep (exceeds default max_depth=5)
        inner = MetadataFilteringCondition(
            conditions=[
                Condition(name="a", comparison_operator="is", value="1"),
            ],
        )
        for _ in range(5):
            inner = MetadataFilteringCondition(
                condition_groups=[inner],
            )
        with pytest.raises(ValueError, match="nesting depth exceeds maximum"):
            _build_metadata_filter_recursive(inner)

    def test_at_max_depth_succeeds(self):
        """Nesting exactly at max_depth boundary should succeed."""
        # Build 5 levels (depth 0..4), which is within default max_depth=5
        inner = MetadataFilteringCondition(
            conditions=[
                Condition(name="a", comparison_operator="is", value="1"),
            ],
        )
        for _ in range(4):
            inner = MetadataFilteringCondition(
                condition_groups=[inner],
            )
        result = _build_metadata_filter_recursive(inner)
        assert result is not None
