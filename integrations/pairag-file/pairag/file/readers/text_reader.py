from pairag.file.readers.base import BaseReader, FileItem, Document, List
from charset_normalizer import from_fp



class TextReader(BaseReader):
    def read(self, file_item: FileItem) -> List[Document]:
        file_item.file.seek(0)
        charset_result = from_fp(file_item.file)

        content = str(charset_result.best())
        return [Document(id_=file_item.id, text=content, metadata=file_item.metadata())]
