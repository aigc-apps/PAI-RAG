from typing import BinaryIO

from loguru import logger
from pairag.file.readers.base import BaseReader, FileItem, Document, List
from pairag.file.utils.split_dataframe import split_dataframe
import chardet


class Csv2MdReader(BaseReader):
    def __init__(
        self, chunk_size: int
    ):
        self.chunk_size = chunk_size
        logger.info(f"Csv2MdReader inited with chunk_size {chunk_size}.")

    def _read_file(self, file: BinaryIO):
        import pandas as pd

        encoding = chardet.detect(file.read(1000))["encoding"]
        file.seek(0)
        encoding = "utf-8"
        if encoding is not None and "GB" in encoding.upper():
            encoding = "GB18030"

        df = pd.read_csv(file, encoding=encoding)
        return df

    def read(self, file_item: FileItem) -> List[Document]:
        """
        Read a CSV file and return a list of Documents.
        """
        df = self._read_file(file_item.file)
        text_list = split_dataframe(df=df, max_tokens=self.chunk_size)
        metadata = file_item.metadata()
        docs = [Document(id_=file_item.id, text=text, metadata=metadata) for text in text_list]

        logger.info(
            f"Successfully read {len(docs)} documents from {file_item.file_name}"
        )
        return docs
