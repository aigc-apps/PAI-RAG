from loguru import logger
from pairag.file.readers.base import BaseReader, FileItem, Document, List
from pairag.file.utils.split_dataframe import split_dataframe
from pairag.file.utils.text_utils import replace_consecutive_spaces

class Json2MdReader(BaseReader):
    def __init__(
        self, chunk_size: int
    ):
        self.chunk_size = chunk_size
        logger.info(f"Json2MdReader inited with chunk_size {chunk_size}.")


    def _read_file(self, file_item: FileItem):
        import pandas as pd

        file_item.file.seek(0)
        df = pd.read_json(file_item.file, lines=True)
        return df

    def read(self, file_item: FileItem) -> List[Document]:
        """
        Read a JSONL file and return a list of Documents.
        """
        df = self._read_file(file_item)
        text_list = split_dataframe(df=df, max_tokens=self.chunk_size)
        for text in text_list:
            text = replace_consecutive_spaces(text)

        metadata = file_item.metadata()
        docs = [Document(id_=file_item.id, text=text, metadata=metadata) for text in text_list]
        logger.info(
            f"Successfully read {len(docs)} documents from {file_item.file_name}"
        )
        return docs
