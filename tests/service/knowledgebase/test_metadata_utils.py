import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

from service.knowledgebase.utils.metadata_utils import validate_metadata_value


class TestValidateMetadataValue:
    def test_string_type_converts(self):
        valid, value = validate_metadata_value(123, "string")
        assert valid is True
        assert value == "123"

    def test_string_type_from_string(self):
        valid, value = validate_metadata_value("hello", "string")
        assert valid is True
        assert value == "hello"

    def test_number_valid_int(self):
        valid, value = validate_metadata_value(42, "number")
        assert valid is True
        assert value == 42

    def test_number_valid_float(self):
        valid, value = validate_metadata_value(3.14, "number")
        assert valid is True
        assert value == 3.14

    def test_number_string_int_converts(self):
        valid, value = validate_metadata_value("42", "number")
        assert valid is True
        assert value == 42
        assert isinstance(value, int)

    def test_number_string_float_converts(self):
        valid, value = validate_metadata_value("3.14", "number")
        assert valid is True
        assert value == 3.14

    def test_number_invalid_string(self):
        valid, value = validate_metadata_value("not_a_number", "number")
        assert valid is False

    def test_datetime_valid_timestamp(self):
        valid, value = validate_metadata_value(1700000000, "datetime")
        assert valid is True
        assert value == 1700000000.0

    def test_datetime_string_timestamp(self):
        valid, value = validate_metadata_value("1700000000", "datetime")
        assert valid is True
        assert value == 1700000000.0

    def test_datetime_invalid_string(self):
        valid, value = validate_metadata_value("not_a_date", "datetime")
        assert valid is False

    def test_unknown_type(self):
        valid, value = validate_metadata_value("test", "unknown_type")
        assert valid is False
