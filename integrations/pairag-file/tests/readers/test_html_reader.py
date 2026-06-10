from bs4 import BeautifulSoup

from pairag.file.readers.html_reader import HtmlReader


def test_table_with_rowspan_and_colspan_stays_rectangular():
    table = BeautifulSoup(
        """
        <table>
            <tr><th rowspan="2">Metric</th><th colspan="2">Values</th></tr>
            <tr><th>Current</th><th>Target</th></tr>
            <tr><td>Latency</td><td>10ms</td><td>8ms</td></tr>
        </table>
        """,
        "html.parser",
    ).find("table")

    pai_table, total_cols = HtmlReader(file_store=None)._convert_table_to_pai_table(table)

    assert total_cols == 3
    assert all(len(row) == total_cols for row in pai_table.data)
    assert pai_table.data[1] == ["Metric", "Current", "Target"]
