"""FAQ parser.

Contains parsers for faq data files.

"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fsspec import AbstractFileSystem

import pandas as pd
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


import chardet
from loguru import logger


class PaiFAQReader(BaseReader):
    r"""FAQ reader.

    Args:

        pandas_config (dict): Options for the `pandas.read_csv` function call.
            Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
            for more information.
            Set to empty dict by default, this means pandas will try to figure
            out the separators, table head, etc. on its own.

    """

    def __init__(
        self,
        *args: Any,
        pandas_config: dict = {},
        format_sheet_data_to_json: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._pandas_config = pandas_config
        self._format_sheet_data_to_json = format_sheet_data_to_json

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse faq file."""
        if fs:
            with fs.open(file) as f:
                encoding = chardet.detect(f.read(100000))["encoding"]
                f.seek(0)
                if encoding is not None and "GB" in encoding.upper():
                    self._pandas_config["encoding"] = "GB18030"
                try:
                    df = pd.read_csv(f, **self._pandas_config)
                except UnicodeDecodeError:
                    logger.info(
                        f"Error: The file {file} encoding could not be decoded."
                    )
                    raise

        else:
            with open(file, "rb") as f:
                encoding = chardet.detect(f.read(100000))["encoding"]
                f.seek(0)
                if encoding is not None and "GB" in encoding.upper():
                    self._pandas_config["encoding"] = "GB18030"
                try:
                    df = pd.read_csv(file, **self._pandas_config)
                except UnicodeDecodeError:
                    logger.info(
                        f"Error: The file {file} encoding could not be decoded."
                    )
                    raise

        docs = [
            Document(
                text=row["question"],
                metadata={
                    **extra_info,
                    "row_number": idx + 1,
                    "faq_answer": row["answer"],
                },
            )
            for idx, row in df.iterrows()
        ]

        return docs
