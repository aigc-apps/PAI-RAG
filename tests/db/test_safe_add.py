import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from unittest.mock import MagicMock, patch
import sqlalchemy as sa


class TestSafeAddColumn:
    @patch("db.op.safe_add.op")
    def test_adds_column_when_not_exists(self, mock_op):
        mock_conn = MagicMock()
        mock_op.get_bind.return_value = mock_conn
        mock_inspector = MagicMock()
        mock_inspector.get_columns.return_value = [{"name": "existing_col"}]

        with patch("db.op.safe_add.sa.inspect", return_value=mock_inspector):
            from db.op.safe_add import safe_add_column

            new_col = sa.Column("new_col", sa.String(50))
            safe_add_column("my_table", new_col)
            mock_op.add_column.assert_called_once()

    @patch("db.op.safe_add.op")
    def test_skips_column_when_exists(self, mock_op):
        mock_conn = MagicMock()
        mock_op.get_bind.return_value = mock_conn
        mock_inspector = MagicMock()
        mock_inspector.get_columns.return_value = [{"name": "existing_col"}]

        with patch("db.op.safe_add.sa.inspect", return_value=mock_inspector):
            from db.op.safe_add import safe_add_column

            existing_col = sa.Column("existing_col", sa.String(50))
            safe_add_column("my_table", existing_col)
            mock_op.add_column.assert_not_called()
