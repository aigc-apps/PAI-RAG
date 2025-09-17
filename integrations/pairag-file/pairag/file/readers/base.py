from abc import ABC, abstractmethod
from typing import List
from llama_index.core.schema import Document

from pairag.file.models.file_item import FileItem


class BaseReader(ABC):
    @abstractmethod
    def read(self, file_item: FileItem) -> List[Document]:
        pass
