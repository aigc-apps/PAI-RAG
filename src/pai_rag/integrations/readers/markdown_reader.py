"""Tabular parser-CSV parser.

Contains parsers for tabular data files.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import re
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import logging

logger = logging.getLogger(__name__)

REGEX_H1 = "=+"
REGEX_H2 = "-+"
REGEX_TABLE_FORMAT = "^\|((:|-)+\|)+$"
REGEX_USELESS_PHRASE = "\{#[0-9a-z]+\}"  # Only for aliyun docs


class MarkdownReader(BaseReader):
    """Markdown file reader."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse csv file.

        Returns:
            Union[str, List[str]]: a string or a List of strings.

        """
        text = ""
        pre_line = ""
        with open(file) as fp:
            line = fp.readline()
            is_code = False
            while line:
                line = re.sub(REGEX_USELESS_PHRASE, "", line)
                striped_line = line.strip()
                if striped_line.startswith("```"):
                    is_code = not is_code

                if not striped_line:
                    text += pre_line
                    pre_line = "\n"
                    line = fp.readline()
                # process table
                elif striped_line.startswith("|") and striped_line.endswith("|"):
                    if re.match(REGEX_TABLE_FORMAT, striped_line):
                        header_line = fp.readline()
                        header_line = re.sub(
                            REGEX_USELESS_PHRASE, "", header_line
                        ).strip()
                        table_headers = [
                            h.strip() for h in header_line.split("|")[1:-1]
                        ]
                    else:
                        table_headers = [
                            h.strip() for h in striped_line.split("|")[1:-1]
                        ]
                        format_line = fp.readline()
                        if not re.match(REGEX_TABLE_FORMAT, format_line):
                            logger.warning(
                                f"Parse markdown table fail: {format_line} is not a table format line."
                            )

                    table_rows = []
                    # read in formatter
                    line = fp.readline()
                    while line and line.startswith("|"):
                        line = re.sub(REGEX_USELESS_PHRASE, "", line).strip()
                        table_rows.append([h.strip() for h in line.split("|")[1:-1]])
                        line = fp.readline()

                    for row in table_rows:
                        for i, col in enumerate(table_headers):
                            text += f"{col}: {row[i]}\n"
                elif re.match(REGEX_H1, striped_line):
                    text += f"# {pre_line}"
                    pre_line = ""
                    line = fp.readline()
                elif re.match(REGEX_H2, striped_line):
                    text += f"## {pre_line}"
                    pre_line = ""
                    line = fp.readline()
                else:
                    text += pre_line
                    pre_line = striped_line
                    if is_code or line.startswith("#") or line.endswith("  \n"):
                        pre_line = f"{striped_line}\n"
                    line = fp.readline()

        text += pre_line

        metadata = {"filename": file.name, "extension": file.suffix}
        if extra_info:
            metadata = {**metadata, **extra_info}

        return [Document(text=text, metadata=metadata)]
