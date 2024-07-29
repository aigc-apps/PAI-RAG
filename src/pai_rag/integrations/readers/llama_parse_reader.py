import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
import logging
import os
import json
import asyncio
import fsspec
from itertools import repeat
from llama_parse import LlamaParse
from functools import reduce
from typing import Callable, Dict, List, Optional
from pathlib import Path
import multiprocessing
import warnings
from llama_index.core.readers.base import BaseReader
from fsspec.implementations.local import LocalFileSystem
from llama_index.core.async_utils import run_jobs, get_asyncio_module
from tqdm import tqdm
from pai_rag.utils.oss_utils import calculate_file_md5

nest_asyncio.apply()

logger = logging.getLogger(__name__)


def is_default_fs(fs: fsspec.AbstractFileSystem) -> bool:
    return isinstance(fs, LocalFileSystem) and not fs.auto_mkdir


def get_default_fs() -> fsspec.AbstractFileSystem:
    return LocalFileSystem()


class LlamaParseDirectoryReader(SimpleDirectoryReader):
    def __init__(
        self,
        input_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        input_files: Optional[List] = None,
        exclude: Optional[List] = None,
        exclude_hidden: bool = True,
        errors: str = "ignore",
        recursive: bool = False,
        encoding: str = "utf-8",
        filename_as_id: bool = False,
        required_exts: Optional[List[str]] = None,
        file_extractor: Optional[Dict[str, BaseReader]] = None,
        num_files_limit: Optional[int] = None,
        file_metadata: Optional[Callable[[str], Dict]] = None,
        raise_on_error: bool = False,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        oss_cache: Optional[object] = None,
    ) -> None:
        """Initialize with parameters.
        Args:
            oss_cache (bool, optional): _description_. Defaults to False.
            api_key (str, optional): _description_. Defaults to None.
        oss_cache (bool): whether open oss cache to store parsed files.
        api_key (str, optional): llama_parse api key.
        """

        super().__init__(
            input_dir=input_dir,
            input_files=input_files,
            exclude=exclude,
            exclude_hidden=exclude_hidden,
            errors=errors,
        )

        """ "markdown" and "text" are available"""
        if not api_key:
            api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            verbose=True,
        )
        self.oss_cache = oss_cache
        self.file_extractor = {".pdf": parser}

    @staticmethod
    def load_file(
        input_file: Path,
        file_metadata: Callable[[str], Dict],
        file_extractor: Dict[str, BaseReader],
        filename_as_id: bool = False,
        oss_cache: object = None,
        encoding: str = "utf-8",
        errors: str = "ignore",
        raise_on_error: bool = False,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> List[Document]:
        """Static method for loading file.

        NOTE: necessarily as a static method for parallel processing.

        Args:
            input_file (Path): _description_
            file_metadata (Callable[[str], Dict]): _description_
            file_extractor (Dict[str, BaseReader]): _description_
            filename_as_id (bool, optional): _description_. Defaults to False.
            oss_cache (cache, optional): _description_. Defaults to None.
            encoding (str, optional): _description_. Defaults to "utf-8".
            errors (str, optional): _description_. Defaults to "ignore".
            fs (Optional[fsspec.AbstractFileSystem], optional): _description_. Defaults to None.

        input_file (Path): File path to read
        file_metadata ([Callable[str, Dict]]): A function that takes
            in a filename and returns a Dict of metadata for the Document.
        file_extractor (Dict[str, BaseReader]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text.
        filename_as_id (bool): Whether to use the filename as the document id.
        oss_cache (Oss_cache): Whether to use the oss_cache.
        encoding (str): Encoding of the files.
            Default is utf-8.
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        raise_on_error (bool): Whether to raise an error if a file cannot be read.
        fs (Optional[fsspec.AbstractFileSystem]): File system to use. Defaults
            to using the local file system. Can be changed to use any remote file system

        Returns:
            List[Document]: loaded documents
        """
        # TODO: make this less redundant

        documents: List[Document] = []
        prefix = "pairag/reader/"
        file_key = None
        if oss_cache:
            file_key = calculate_file_md5(input_file, prefix)
        if oss_cache and oss_cache.get_object(file_key):
            bytes_content = oss_cache.get_object(file_key)
            documents_json = bytes_content.read().decode("utf-8")
            documents_data = json.loads(documents_json)
            documents = [Document(**doc_data) for doc_data in documents_data]

        else:
            default_file_reader_cls = SimpleDirectoryReader.supported_suffix_fn()
            default_file_reader_suffix = list(default_file_reader_cls.keys())
            metadata: Optional[dict] = None

            if file_metadata is not None:
                metadata = file_metadata(str(input_file))

            file_suffix = input_file.suffix.lower()
            if (
                file_suffix in default_file_reader_suffix
                or file_suffix in file_extractor
            ):
                # use file readers
                if file_suffix not in file_extractor:
                    # instantiate file reader if not already
                    reader_cls = default_file_reader_cls[file_suffix]
                    file_extractor[file_suffix] = reader_cls()
                reader = file_extractor[file_suffix]

                # load data -- catch all errors except for ImportError
                try:
                    kwargs = {"extra_info": metadata}
                    if fs and not is_default_fs(fs):
                        kwargs["fs"] = fs
                    docs = reader.load_data(input_file, **kwargs)
                except ImportError as e:
                    # ensure that ImportError is raised so user knows
                    # about missing dependencies
                    raise ImportError(str(e))
                except Exception as e:
                    if raise_on_error:
                        raise Exception("Error loading file") from e
                    # otherwise, just skip the file and report the error
                    print(
                        f"Failed to load file {input_file} with error: {e}. Skipping...",
                        flush=True,
                    )
                    return []

                # iterate over docs if needed
                if filename_as_id:
                    for i, doc in enumerate(docs):
                        doc.id_ = f"{input_file!s}_part_{i}"

                documents.extend(docs)
            else:
                # do standard read
                fs = fs or get_default_fs()
                with fs.open(input_file, errors=errors, encoding=encoding) as f:
                    data = f.read().decode(encoding, errors=errors)

                doc = Document(text=data, metadata=metadata or {})
                if filename_as_id:
                    doc.id_ = str(input_file)

                documents.append(doc)
            if oss_cache:
                logger.info("upload file")

                def document_serializer(obj):
                    if isinstance(obj, Document):
                        return obj.__dict__
                    """ 或者返回一个自定义字典，具体取决于 Document 的结构
                        对于不是 Document 类型的，可以抛出异常
                    """
                    raise TypeError(f"Type {type(obj)} not serializable")

                json_data = json.dumps(documents, default=document_serializer)
                utf8_encoded_data = json_data.encode("utf-8")
                oss_cache.put_object(file_key, utf8_encoded_data)

        return documents

    def load_data(
        self,
        show_progress: bool = False,
        num_workers: Optional[int] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
            num_workers  (Optional[int]): Number of workers to parallelize data-loading over.
            fs (Optional[fsspec.AbstractFileSystem]): File system to use. If fs was specified
                in the constructor, it will override the fs parameter here.

        Returns:
            List[Document]: A list of documents.
        """
        documents = []

        files_to_process = self.input_files
        fs = fs or self.fs

        if num_workers and num_workers > 1:
            if num_workers > multiprocessing.cpu_count():
                warnings.warn(
                    "Specified num_workers exceed number of CPUs in the system. "
                    "Setting `num_workers` down to the maximum CPU count."
                )
            with multiprocessing.get_context("spawn").Pool(num_workers) as p:
                results = p.starmap(
                    LlamaParseDirectoryReader.load_file,
                    zip(
                        files_to_process,
                        repeat(self.file_metadata),
                        repeat(self.file_extractor),
                        repeat(self.filename_as_id),
                        repeat(self.oss_cache),
                        repeat(self.encoding),
                        repeat(self.errors),
                        repeat(self.raise_on_error),
                        repeat(fs),
                    ),
                )
                documents = reduce(lambda x, y: x + y, results)

        else:
            if show_progress:
                files_to_process = tqdm(
                    self.input_files, desc="Loading files", unit="file"
                )
            for input_file in files_to_process:
                documents.extend(
                    LlamaParseDirectoryReader.load_file(
                        input_file=input_file,
                        file_metadata=self.file_metadata,
                        file_extractor=self.file_extractor,
                        oss_cache=self.oss_cache,
                        filename_as_id=self.filename_as_id,
                        encoding=self.encoding,
                        errors=self.errors,
                        raise_on_error=self.raise_on_error,
                        fs=fs,
                    )
                )

        return self._exclude_metadata(documents)

    async def aload_data(
        self,
        show_progress: bool = False,
        num_workers: Optional[int] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
            num_workers  (Optional[int]): Number of workers to parallelize data-loading over.
            fs (Optional[fsspec.AbstractFileSystem]): File system to use. If fs was specified
                in the constructor, it will override the fs parameter here.

        Returns:
            List[Document]: A list of documents.
        """
        files_to_process = self.input_files
        fs = fs or self.fs

        coroutines = [self.aload_file(input_file) for input_file in files_to_process]
        if num_workers:
            document_lists = await run_jobs(
                coroutines, show_progress=show_progress, workers=num_workers
            )
        elif show_progress:
            _asyncio = get_asyncio_module(show_progress=show_progress)
            document_lists = await _asyncio.gather(*coroutines)
        else:
            document_lists = await asyncio.gather(*coroutines)
        documents = [doc for doc_list in document_lists for doc in doc_list]

        return self._exclude_metadata(documents)
