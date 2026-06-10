from bs4 import BeautifulSoup

from pairag.file.readers.html_reader import HtmlReader


def _convert_html_table(html: str):
    table = BeautifulSoup(html, "html.parser").find("table")
    return HtmlReader(file_store=None)._convert_table_to_pai_table(table)


def test_table_with_rowspan_and_colspan_stays_rectangular():
    pai_table, total_cols = _convert_html_table(
        """
        <table>
            <tr><th rowspan="2">Metric</th><th colspan="2">Values</th></tr>
            <tr><th>Current</th><th>Target</th></tr>
            <tr><td>Latency</td><td>10ms</td><td>8ms</td></tr>
        </table>
        """
    )
    assert total_cols == 3
    assert all(len(row) == total_cols for row in pai_table.data)
    assert pai_table.data[1] == ["Metric", "Current", "Target"]


def test_empty_table_converts_to_empty_row():
    pai_table, total_cols = _convert_html_table("<table></table>")

    assert total_cols == 0
    assert pai_table.data == [[]]


def test_colspan_only_table_stays_rectangular():
    pai_table, total_cols = _convert_html_table(
        """
        <table>
            <tr><th colspan="2">Name</th><th>Score</th></tr>
            <tr><td>Alice</td><td>A.</td><td>10</td></tr>
        </table>
        """
    )

    assert total_cols == 3
    assert pai_table.data == [
        ["Name", "Name", "Score"],
        ["Alice", "A.", "10"],
    ]


def test_later_row_can_expand_table_width():
    pai_table, total_cols = _convert_html_table(
        """
        <table>
            <tr><th>Name</th></tr>
            <tr><td>Alice</td><td>10</td><td>Active</td></tr>
        </table>
        """
    )

    assert total_cols == 3
    assert pai_table.data == [
        ["Name", "", ""],
        ["Alice", "10", "Active"],
    ]


def test_malformed_table_is_normalized_by_parser():
    pai_table, total_cols = _convert_html_table(
        """
        <table>
            <tr><th>Name<th>Score
            <tr><td>Alice<td>10
        </table>
        """
    )

    assert total_cols > 0
    assert all(len(row) == total_cols for row in pai_table.data)
    assert "Name" in pai_table.data[0][0]
