from loguru import logger
from pairag.mcp.rag.file.readers.base import BaseReader, FileItem, Document, List


class JsonReader(BaseReader):
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
        text_list = [
            "\n".join([f"{k}:{v}" for k, v in record.items()])
            for record in df.to_dict("records")
        ]

        metadata = file_item.metadata()
        docs = [Document(text=text, metadata=metadata) for text in text_list]
        logger.info(
            f"Successfully read {len(docs)} documents from {file_item.file_name}"
        )
        return docs
