"""Html parser.

Parser for html files.

"""

from pathlib import Path
from fsspec import AbstractFileSystem
from typing import Any, Dict, List, Optional
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from pai_rag.integrations.readers.html.utils.filter import filter_html
from pai_rag.integrations.readers.html.utils.split import split_html

DEFAULT_RANK_LABEL = "h2"


class HtmlReader(BaseReader):
    """Html parser.

    Extract text from HTML file.
    Returns dictionary with keys as headers and values as the text between headers.

    """

    def __init__(
        self,
        *args: Any,
        rank_label: str = DEFAULT_RANK_LABEL,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self.rank_label = rank_label

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        with open(file) as f:
            html_data = f.read()

        header, context = filter_html(html_data)
        context_with_h1 = [header + "\n"] + context
        subdocs = split_html(context_with_h1, self.rank_label)

        metadata = extra_info or {}
        metadata["header"] = header
        metadata["file_type"] = "HTML"
        docs = [Document(text=doc, metadata=metadata) for doc in subdocs]

        return docs
