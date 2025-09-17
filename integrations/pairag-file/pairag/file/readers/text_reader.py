from pairag.file.readers.base import BaseReader, FileItem, Document, List


class TextReader(BaseReader):
    def read(self, file_item: FileItem) -> List[Document]:
        file_item.file.seek(0)
        content = file_item.file.read().decode("utf-8")
        return [Document(id_=file_item.id, text=content, metadata=file_item.metadata())]
