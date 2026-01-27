from typing import BinaryIO
from pairag.file.readers.base import BaseReader, FileItem, Document, List
from charset_normalizer import detect
from pairag.file.utils.text_utils import replace_consecutive_spaces
from loguru import logger

def smart_read(file: BinaryIO) -> str:
    file.seek(0)
    encoding = detect(file.read(10240)).get("encoding")
    if encoding:
        logger.info(f"Detected encoding: {encoding}")
        file.seek(0)
        return file.read().decode(encoding)
    
    encodings = ["utf-8-sig", "gb18030", "utf-16"]
    for encoding in encodings:
        try:
            logger.info(f"Trying to decode file with encoding: {encoding}")
            file.seek(0)
            return file.read().decode(encoding)
        except Exception as e:
            logger.warning(f"Failed to decode file with encoding {encoding}: {e}")
            continue

    logger.error("Failed to detect encoding, using utf-8")
    return file.read().decode("utf-8")


class TextReader(BaseReader):
    def read(self, file_item: FileItem) -> List[Document]:
        content = replace_consecutive_spaces(smart_read(file_item.file))
        if not content:
            logger.warning(f"Parse file {file_item.file_name} into empty content.")
            return []
        return [Document(id_=file_item.id, text=content, metadata=file_item.metadata())]
