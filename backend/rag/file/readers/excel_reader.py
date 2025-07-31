from loguru import logger
from rag.file.readers.base import BaseReader, FileItem, Document, List


class ExcelReader(BaseReader):
    def _read_file(self, file_item: FileItem):
        import pandas as pd

        if file_item.file_extension == ".xls":
            read_engine = "xlrd"
        else:
            read_engine = "openpyxl"

        df = pd.read_excel(file_item.file, engine=read_engine)
        return df

    def read(self, file_item: FileItem) -> List[Document]:
        """
        Read a Excel file and return a list of Documents.
        """
        df = self._read_file(file_item)
        text_list = [
            "\n".join([f"{k}:{v}" for k, v in record.items()])
            for record in df.to_dict("records")
        ]

        metadata = file_item.metadata()
        docs = [Document(id_=file_item.id, text=text, metadata=metadata) for text in text_list]
        logger.info(
            f"Successfully read {len(docs)} documents from {file_item.file_name}"
        )

        return docs
