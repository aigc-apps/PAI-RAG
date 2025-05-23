"""Tabular parser-Excel parser.

Contains parsers for tabular data files.

"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PaiJsonLReader(BaseReader):
    """JsonL reader."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)

    def load_data(
        self,
        file_path: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as file:
            json_lines = [line.strip() for line in file.readlines()]

        file_name = os.path.basename(file_path)
        extra_info = extra_info or {}
        extra_info["file_path"] = str(file_path)
        extra_info["file_name"] = file_name

        docs = []
        for i, text in enumerate(json_lines):
            extra_info["row_number"] = i + 1
            docs.append(Document(text=text, metadata=extra_info))
        return docs
