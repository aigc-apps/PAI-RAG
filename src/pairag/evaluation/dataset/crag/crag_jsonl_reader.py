"""Tabular parser-Excel parser.

Contains parsers for tabular data files.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import json


class CragJsonLReader(BaseReader):
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

        docs = []
        for i, text in enumerate(json_lines):
            json_data = json.loads(text)
            search_results = json_data["search_results"]
            for j, search_result in enumerate(search_results):
                extra_info["row_number"] = i + 1
                extra_info["dataset_source"] = "crag"
                docs.append(
                    Document(
                        doc_id=f"{json_data['interaction_id']}__{j}",
                        text=search_result["page_snippet"],
                        metadata=extra_info,
                    )
                )
        return docs
