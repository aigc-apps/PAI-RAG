import asyncio
import dataclasses
import functools
import os
import re
import time
from dataclasses import field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import fsspec
import psutil
from fsspec import AbstractFileSystem
from llama_index.core import SimpleDirectoryReader
from llama_index.core.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import (
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
)
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.node_parser.text.sentence import DEFAULT_PARAGRAPH_SEP
from llama_index.core.readers.file.base import BaseReader, default_file_metadata_func
from llama_index.core.schema import BaseNode, TextNode, TransformComponent
from pairag.file.nodeparsers.pai.pai_markdown_parser import MarkdownNodeParser
from pairag.file.readers.pai.file_readers.pai_csv_reader import PaiPandasCSVReader
from pairag.file.readers.pai.file_readers.pai_docx_reader import PaiDocxReader
from pairag.file.readers.pai.file_readers.pai_excel_reader import PaiPandasExcelReader
from pairag.file.readers.pai.file_readers.pai_html_reader import PaiHtmlReader
from pairag.file.readers.pai.file_readers.pai_jsonl_reader import PaiJsonLReader
from pairag.file.readers.pai.file_readers.pai_markdown_reader import PaiMarkdownReader
from pairag.file.readers.pai.file_readers.pai_pdf_reader import PaiPDFReader
from pairag.file.readers.pai.file_readers.pai_pptx_reader import PaiPptxReader
from pairag.file.store.pai_image_store import PaiImageStore
from pydantic import Field

# Simplified constant definitions
DEFAULT_CHUNK_NODES_DIR = ".artifacts/chunk_nodes"
DEFAULT_CHUNKING_BATCH_FILE_SIZE = 200 * 1024 * 1024
DEFAULT_CHUNKING_BATCH_NUM = 10000
DEFAULT_NODE_SOURCE_FIELD = "source"
DEFAULT_MODIFIED_AT_FIELD = "modified_at"
DEFAULT_MD5_FIELD = "md5"

DOC_TYPES_DO_NOT_NEED_CHUNKING = set([".csv", ".xlsx", ".xls", ".jsonl"])
DOC_TYPES_CONVERT_TO_MD = set([".md", ".pdf", ".docx", ".htm", ".html", ".pptx"])
DEFAULT_WINDOW_SIZE = 3


def get_logger(name: str):
    """Get logger instance"""
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = get_logger("text_chunking")


def generate_md5_hash(file_path: str) -> str:
    """Generate MD5 hash for a file"""
    import hashlib
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@dataclasses.dataclass
class ChunkArgs:
    input_path: str = field(
        metadata={"help": "Input path for data to be chunked."}
    )
    output_path: str = field(metadata={"help": "Output path for chunked data."})

    chunk_size: int = field(
        default=DEFAULT_CHUNK_SIZE,
        metadata={"help": "Chunk size for document splitting"},
    )
    chunk_overlap: int = field(
        default=DEFAULT_CHUNK_OVERLAP,
        metadata={"help": "Chunk overlap size for document splitting"},
    )
    paragraph_separator: str = field(
        default=DEFAULT_PARAGRAPH_SEP,
        metadata={
            "help": "Separator used for paragraph splitting. Default is three newlines."
        },
    )
    concat_row: bool = field(
        default=False,
        metadata={"help": "Whether to concatenate rows when reading csv/excel files"},
    )
    recursive: bool = field(
        default=True,
        metadata={"help": "Whether to recursively read directories"},
    )
    enable_image_ocr: bool = field(
        default=False,
        metadata={"help": "Whether to enable image OCR when reading PDF files"},
    )
    node_parser_type: str = field(
        default="Sentence",
        metadata={
            "help": "Node parser type used for chunking. Options: [Token, Sentence, SentenceWindow]"
        },
    )
    support_file_exts: List[str] = field(
        default_factory=lambda: [
            ".html",
            ".htm",
            ".pdf",
            ".csv",
            ".xlsx",
            ".xls",
            ".txt",
            ".docx",
            ".pptx",
            ".doc",
            ".ppt",
            ".md",
            ".jsonl",
        ],
        metadata={"help": "Supported file extensions."},
    )

    def __post_init__(self):
        if not self.node_parser_type:
            self.node_parser_type = "Sentence"
        elif self.node_parser_type.lower() not in [
            "token",
            "sentence",
            "sentenceWindow",
        ]:
            raise ValueError(
                f"Invalid node_parser_type: {self.node_parser_type}. "
                f"Options: [Token, Sentence, SentenceWindow]"
            )

        if not os.path.isdir(self.input_path):
            raise ValueError(f"Input path {self.input_path} is not a directory.")


class LocalImageStore(PaiImageStore):
    """Local image storage for storing images from documents"""

    def __init__(self, tmp_local_dir: str):
        self.tmp_local_dir = tmp_local_dir
        os.makedirs(tmp_local_dir, exist_ok=True)

    def upload_image(self, image, doc_name: str):
        """Upload image to local storage"""
        import hashlib
        import io

        # Generate image filename
        if hasattr(image, 'tobytes'):
            image_bytes = image.tobytes()
        else:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

        image_hash = hashlib.md5(image_bytes).hexdigest()
        image_filename = f"{doc_name}_{image_hash}.png"
        image_path = os.path.join(self.tmp_local_dir, image_filename)

        # Save image
        if hasattr(image, 'save'):
            image.save(image_path, 'PNG')
        else:
            with open(image_path, 'wb') as f:
                f.write(image_bytes)

        return image_path


class CustomNodeParser(TransformComponent):
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="Token size for each chunk.",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=None,
        description="Token overlap for each chunk during splitting.",
        gte=0,
    )
    paragraph_separator: str = Field(
        default=DEFAULT_PARAGRAPH_SEP, description="Separator between paragraphs."
    )

    node_parser_type: str = Field(
        default="Sentence",
        description="Node parser type used for chunking. Options: [Token, Sentence, SentenceWindow]",
    )

    def __init__(
        self,
        node_parser_type: str,
        chunk_size: int,
        chunk_overlap: int,
        paragraph_separator: str,
    ):
        super().__init__(
            node_parser_type=node_parser_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator=paragraph_separator,
        )
        logger.info(
            f"CustomNodeParser initialized, node_parser_type: {node_parser_type}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}, paragraph_separator: {paragraph_separator}"
        )

    @classmethod
    def from_chunk_args(cls, chunk_args: ChunkArgs):
        return cls(
            node_parser_type=chunk_args.node_parser_type,
            chunk_size=chunk_args.chunk_size,
            chunk_overlap=chunk_args.chunk_overlap,
            paragraph_separator=chunk_args.paragraph_separator,
        )

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        return self.process(nodes, **kwargs)

    def process(self, docs: List[BaseNode], **kwags) -> List[BaseNode]:
        node_parser = self._init_node_parser(
            self.node_parser_type,
            self.chunk_size,
            self.chunk_overlap,
            self.paragraph_separator,
        )

        nodes = []
        for idx, doc in enumerate(docs):
            file_name = doc.metadata.get("file_name", "dummy.txt")
            logger.info(f"Starting to chunk document: {file_name}")
            doc_type = os.path.splitext(file_name)[1].lower()
            if doc_type in DOC_TYPES_DO_NOT_NEED_CHUNKING:
                # Table format documents
                node_id = default_id_func(idx, doc)
                nodes.append(
                    TextNode(id_=node_id, text=doc.text, metadata=doc.metadata)
                )
            elif doc_type in DOC_TYPES_CONVERT_TO_MD:
                # Convert to markdown format documents (pdf, docx, htm, html, pptx)
                md_node_parser = MarkdownNodeParser(
                    id_func=default_id_func,
                    chunk_size=self.chunk_size,
                    chunk_overlap_size=self.chunk_overlap,
                    base_parser=node_parser,
                )
                parsed_nodes = md_node_parser.get_nodes_from_documents([doc])
                nodes.extend(self._post_process_nodes(parsed_nodes))
            else:
                # Text format documents
                nodes.extend(node_parser.get_nodes_from_documents([doc]))
        logger.info(
            f"[CustomNodeParser] Split {len(docs)} documents into {len(nodes)} nodes."
        )
        return nodes

    def _init_node_parser(
        self,
        node_parser_type: str,
        chunk_size: int,
        chunk_overlap: int,
        paragraph_separator: str,
    ):
        if node_parser_type.lower() == "token":
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif node_parser_type.lower() == "sentence":
            return SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                paragraph_separator=paragraph_separator,
            )
        elif node_parser_type.lower() == "sentencewindow":
            return SentenceWindowNodeParser(
                sentence_splitter=SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    paragraph_separator=paragraph_separator,
                ).split_text,
                window_size=DEFAULT_WINDOW_SIZE,
            )
        else:
            raise ValueError(
                f"Invalid node_parser_type: {node_parser_type}. "
                f"Options: [Token, Sentence, SentenceWindow]"
            )

    def _post_process_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        Post-process nodes, including:
        1. Handle image references, replace OSS URI with placeholders
        2. Add page_label and other metadata
        """
        processed_nodes = []
        for i, node in enumerate(nodes):
            # Add page_label, maintain MarkdownNodeParser order; compatible naming, actually chunk_label
            new_metadata = node.metadata.copy()
            if "page_label" not in new_metadata:
                new_metadata["page_label"] = str(i + 1)  # Start from 1

            # Handle image references
            image_refs = self._extract_image_references(node.text)
            image_metadata = {}
            processed_text = node.text
            if image_refs:
                logger.info(
                    f"Node id: {node.id_} has {len(image_refs)} image references."
                )
                for j, (alt_text, image_uri) in enumerate(image_refs):
                    placeholder = f"IMAGE_PLACEHOLDER_{j}"
                    # Replace image references with placeholders in text
                    pattern = (
                        re.escape(f"![{alt_text}]({image_uri})")
                        if alt_text
                        else re.escape(f"![]({image_uri})")
                    )
                    replacement = (
                        f"![{alt_text}]({placeholder})"
                        if alt_text
                        else f"![]({placeholder})"
                    )
                    processed_text = re.sub(
                        pattern, replacement, processed_text, count=1
                    )
                    image_metadata[placeholder] = {
                        "alt_text": alt_text,
                        "oss_uri": image_uri,
                    }

            new_metadata["images"] = image_metadata
            new_node = TextNode(
                id_=node.id_, text=processed_text, metadata=new_metadata
            )
            processed_nodes.append(new_node)

        logger.info(
            f"Post-processed {len(processed_nodes)} nodes, added page_label and image references."
        )
        return processed_nodes

    def _extract_image_references(self, text: str) -> List[tuple]:
        """
        Extract all image references from text, return [(alt_text, image_uri), ...]
        """
        pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
        matches = re.findall(pattern, text)
        image_refs = []
        for alt_text, uri in matches:
            if uri.startswith("oss://"):
                image_refs.append((alt_text.strip(), uri.strip()))
        return image_refs


def extract_metadata(
    file_path: str,
    fs: Optional[fsspec.AbstractFileSystem] = None,
) -> Dict:
    """Extract file metadata"""
    metadata = default_file_metadata_func(file_path=file_path, fs=fs)
    metadata[DEFAULT_NODE_SOURCE_FIELD] = file_path

    stat = os.stat(file_path)
    metadata[DEFAULT_MODIFIED_AT_FIELD] = stat.st_mtime
    metadata[DEFAULT_MD5_FIELD] = generate_md5_hash(file_path)

    return metadata


def get_file_metadata_func() -> Callable:
    """Get file metadata function"""
    return functools.partial(extract_metadata)


def custom_file_reader(
    concat_row: bool, enable_image_ocr: bool, output_path: str
) -> Dict[str, BaseReader]:
    """Custom file reader using pairag-file library"""
    image_store = LocalImageStore(output_path)

    file_readers = {
        ".html": PaiHtmlReader(
            image_store=image_store,  # Store html images
        ),
        ".htm": PaiHtmlReader(
            image_store=image_store,  # Store html images
        ),
        ".doc": PaiDocxReader(
            image_store=image_store,  # Store doc images
        ),
        ".ppt": PaiPptxReader(
            image_store=image_store,  # Store ppt images
        ),
        ".docx": PaiDocxReader(
            image_store=image_store,  # Store docx images
        ),
        ".pdf": PaiPDFReader(
            enable_mandatory_ocr=enable_image_ocr,
            image_store=image_store,  # Store pdf images
            model_dir="ml/model",  # Default model directory
        ),
        ".pptx": PaiPptxReader(
            image_store=image_store,  # Store pptx images
        ),
        ".md": PaiMarkdownReader(
            image_store=image_store,  # Store markdown images
        ),
        ".csv": PaiPandasCSVReader(
            concat_rows=concat_row,
        ),
        ".xlsx": PaiPandasExcelReader(
            concat_rows=concat_row,
        ),
        ".xls": PaiPandasExcelReader(
            concat_rows=concat_row,
        ),
        ".jsonl": PaiJsonLReader(),
    }
    return file_readers


def get_proper_batch_size(
    max_bytes_per_batch: int = DEFAULT_CHUNKING_BATCH_FILE_SIZE,
) -> int:
    """Get appropriate batch size based on system available memory"""
    available_memory = psutil.virtual_memory().available
    # Use 1/4 of available memory as safety threshold
    dynamic_limit = available_memory // 4

    return min(dynamic_limit, max_bytes_per_batch)


class CustomSimpleDirectoryReader(SimpleDirectoryReader):
    """Custom SimpleDirectoryReader to improve performance when processing large numbers of files"""

    def __init__(
        self,
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
        fs: Optional[AbstractFileSystem] = None,
    ) -> None:
        """
        Custom SimpleDirectoryReader implementation to improve performance when processing large numbers of files.

        This implementation bypasses expensive file validation steps in the parent class by only passing one file
        to the parent constructor. This helps avoid expensive network I/O when processing large input lists.
        Then manually assign the complete file list as Path objects.
        """

        if not input_files:
            raise ValueError("Must provide `input_files`.")

        # Only pass one file to avoid complete isfile check in parent class
        super().__init__(
            input_dir=None,
            input_files=input_files[:1],
            exclude=exclude,
            exclude_hidden=exclude_hidden,
            errors=errors,
            recursive=recursive,
            encoding=encoding,
            filename_as_id=filename_as_id,
            required_exts=required_exts,
            file_extractor=file_extractor,
            num_files_limit=num_files_limit,
            file_metadata=file_metadata,
            raise_on_error=raise_on_error,
            fs=fs,
        )

        # Override with real files
        if input_files:
            self.input_files = [Path(path) for path in input_files]
            if num_files_limit and len(self.input_files) > num_files_limit:
                self.input_files = self.input_files[:num_files_limit]
                logger.warning(
                    f"File count {len(self.input_files)} exceeds limit {num_files_limit}. "
                    f"Only using first {num_files_limit} files."
                )


def chunk_files(input_files: List[str], chunk_args: ChunkArgs) -> List[BaseNode]:
    """
    Chunk files into nodes.
    """
    if not input_files:
        return []

    total_file_size = sum(os.path.getsize(f) for f in input_files)

    # Use CustomSimpleDirectoryReader instead of SimpleDirectoryReader to skip internal iteration
    # input_files and file validation. This can significantly improve performance when processing large numbers of files.
    logger.info("Initializing SimpleDirectorReader")
    data_reader = CustomSimpleDirectoryReader(
        input_files=input_files,
        file_extractor=custom_file_reader(
            concat_row=chunk_args.concat_row,
            enable_image_ocr=chunk_args.enable_image_ocr,
            output_path=chunk_args.output_path,
        ),
        raise_on_error=False,
        file_metadata=get_file_metadata_func(),
    )

    logger.info(f"Starting to load data, total {len(input_files)} files")
    num_workers = min(os.cpu_count(), 32)
    docs = data_reader.load_data(
        show_progress=False,
        num_workers=num_workers,
    )
    logger.info(
        f"Data loading completed, total files: {len(docs)}, total file size: {total_file_size}"
    )
    logger.info("Starting parsing and chunking")

    pipeline = IngestionPipeline(
        transformations=[
            CustomNodeParser.from_chunk_args(chunk_args),
        ],
        disable_cache=True,
    )
    loop = asyncio.get_event_loop()
    nodes = loop.run_until_complete(
        pipeline.arun(
            documents=docs,
            num_workers=os.cpu_count(),
        )
    )
    logger.info(f"Chunking completed, total {len(nodes)} nodes obtained")
    return nodes


def get_files_to_process(input_path: str, support_file_exts: List[str]) -> List[str]:
    """Get list of files to process"""
    files_to_process = []

    for root, dirs, files in os.walk(input_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in support_file_exts:
                files_to_process.append(file_path)

    return files_to_process


def save_nodes_to_jsonl(nodes: List[BaseNode], output_path: str):
    """Save nodes to JSONL format"""
    import json

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for node in nodes:
            node_dict = {
                'id': node.id_,
                'text': node.text,
                'metadata': node.metadata
            }
            f.write(json.dumps(node_dict, ensure_ascii=False) + '\n')


def print_chunks_stats(nodes: List[BaseNode], total_time_consumed: float = 0.0):
    """Print chunking statistics"""
    from collections import Counter

    # Count chunks per source file
    source_counts = Counter()
    for node in nodes:
        source = node.metadata.get(DEFAULT_NODE_SOURCE_FIELD, "unknown")
        source_counts[source] += 1

    total_files = len(source_counts)
    total_chunks = len(nodes)
    avg_chunks_per_file = total_chunks / total_files if total_files > 0 else 0

    logger.info("Chunking statistics:")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Total chunks: {total_chunks}")
    logger.info(f"  Average chunks per file: {avg_chunks_per_file:.2f}")
    logger.info(f"  Total time consumed: {total_time_consumed:.2f} seconds")

    # Show chunks count for top 10 files
    logger.info("  Top 10 files by chunk count:")
    for source, count in source_counts.most_common(10):
        logger.info(f"    {source}: {count} chunks")


def print_system_info():
    """Print system information"""
    logger.info("System information:")
    logger.info(f"  CPU cores: {os.cpu_count()}")
    logger.info(f"  Memory usage: {psutil.virtual_memory().percent}%")
    logger.info(f"  Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")


def main(chunk_args: ChunkArgs):
    """Main function"""
    logger.info(f"Document chunking parameters: {chunk_args}")
    begin = time.time()

    if not os.path.exists(chunk_args.input_path):
        raise ValueError(
            f"Invalid input_path {chunk_args.input_path}, path does not exist."
        )

    if not os.path.exists(chunk_args.output_path):
        os.makedirs(chunk_args.output_path, exist_ok=True)

    if os.path.isdir(chunk_args.output_path):
        chunk_output_path = os.path.join(
            chunk_args.output_path, DEFAULT_CHUNK_NODES_DIR
        )
    else:
        raise ValueError(
            f"Invalid output_path {chunk_args.output_path}, should be a directory."
        )

    print_system_info()

    # Get files to process
    files_to_process = get_files_to_process(chunk_args.input_path, chunk_args.support_file_exts)
    logger.info(f"Found {len(files_to_process)} files to process")

    if not files_to_process:
        logger.warning("No files found to process")
        return

    # Adjust batch size based on system memory
    max_bytes_per_batch = get_proper_batch_size(DEFAULT_CHUNKING_BATCH_FILE_SIZE)
    logger.info(f"Set max_bytes_per_batch to {max_bytes_per_batch} based on system available memory")

    # Simple batch processing logic
    batch_size = DEFAULT_CHUNKING_BATCH_NUM
    total_nodes = []

    for i in range(0, len(files_to_process), batch_size):
        batch_files = files_to_process[i:i + batch_size]
        batch_size_bytes = sum(os.path.getsize(f) for f in batch_files)

        # If batch size exceeds limit, further split
        if batch_size_bytes > max_bytes_per_batch:
            sub_batches = []
            current_batch = []
            current_size = 0

            for file_path in batch_files:
                file_size = os.path.getsize(file_path)
                if current_size + file_size > max_bytes_per_batch and current_batch:
                    sub_batches.append(current_batch)
                    current_batch = [file_path]
                    current_size = file_size
                else:
                    current_batch.append(file_path)
                    current_size += file_size

            if current_batch:
                sub_batches.append(current_batch)
        else:
            sub_batches = [batch_files]

        # Process each sub-batch
        for sub_batch in sub_batches:
            nodes = chunk_files(sub_batch, chunk_args)
            total_nodes.extend(nodes)

            # Save nodes to file
            output_file_path = os.path.join(
                chunk_output_path, f"chunk_nodes_{len(total_nodes)}.jsonl"
            )
            save_nodes_to_jsonl(nodes, output_file_path)

            logger.info(
                f"Progress: {len(total_nodes)} nodes processed. Chunks saved to: {output_file_path}"
            )

    end = time.time()
    print_chunks_stats(total_nodes, int(end - begin))
    logger.info(f"Document chunking completed, total time: {end - begin:.2f} seconds")


if __name__ == "__main__":
    # Simple command line argument parsing
    import argparse

    parser = argparse.ArgumentParser(description="Document chunking tool")
    parser.add_argument("--input_path", required=True, help="Input path")
    parser.add_argument("--output_path", required=True, help="Output path")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap")
    parser.add_argument("--node_parser_type", default="Sentence", help="Node parser type")
    parser.add_argument("--concat_row", action="store_true", help="Concatenate rows")
    parser.add_argument("--enable_image_ocr", action="store_true", help="Enable image OCR")

    args = parser.parse_args()

    chunk_args = ChunkArgs(
        input_path=args.input_path,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        node_parser_type=args.node_parser_type,
        concat_row=args.concat_row,
        enable_image_ocr=args.enable_image_ocr,
    )

    main(chunk_args)
