from llama_index.core.node_parser.interface import TextSplitter
from typing import (
    List,
    Callable,
    Sequence,
    Any,
)
from llama_index.core.schema import BaseNode
from pairag.file.utils.tokenization import get_tokenizer
from llama_index.core.bridge.pydantic import Field
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
)
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from pairag.file.utils.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_PARAGRAPH_SEPARATOR
from pairag.file.utils.tokenization import estimate_tokens_in_text
import re
from loguru import logger

class ParagraphSplitter(TextSplitter):
    """Splitting text using paragraph separator."""

    paragraph_separator: str = Field(default="\n\n", description="Separator between paragraphs.")
    chunk_size: int = Field(default=1024, description="chunk size.")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size.")

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        paragraph_separator: str = DEFAULT_PARAGRAPH_SEPARATOR,
        id_func: Callable[[int, BaseNode], str] = None,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, paragraph_separator=paragraph_separator)
        self.id_func = id_func
        self._tokenizer = get_tokenizer()



    def split_text(self, text: str) -> tuple[list[str], list[int]]:
        """Split text into chunks."""
        logger.info(f"Splitting text into chunks with chunk size {self.chunk_size}")
        return self._split_text(text, paragraph_separator=self.paragraph_separator, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def _split_text_with_regex(self, text: str, paragraph_separator: str = None) -> list[str]:
        if paragraph_separator is None:
            paragraph_separator = DEFAULT_PARAGRAPH_SEPARATOR
        splits = re.split(paragraph_separator, text)
        return [s for s in splits if (s not in {"", "\n"})]


    def _split_text_on_tokens(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> tuple[list[str], list[int]]:
        """Split incoming text and return chunks using tokenizer."""
        splits: list[str] = []
        token_counts: list[int] = []
        
        if chunk_size is None:
            chunk_size = DEFAULT_CHUNK_SIZE
        
        if chunk_overlap is None:
            chunk_overlap = DEFAULT_CHUNK_OVERLAP
        
        result = self._tokenizer(
            text, 
            return_offsets_mapping=True, 
            return_attention_mask=False, 
            add_special_tokens=False
        )
        input_ids = result["input_ids"]
        offset_mapping = result["offset_mapping"]
        
        start_idx = 0
        while start_idx < len(input_ids):
            chunk_offset_mapping = offset_mapping[start_idx:start_idx + chunk_size]
            
            if chunk_offset_mapping:
                chunk_start_char = chunk_offset_mapping[0][0]
                chunk_end_char = chunk_offset_mapping[-1][1]
                # Extract text using character offsets to avoid splitting characters
                chunk_text = text[chunk_start_char:chunk_end_char]
                splits.append(chunk_text)
                token_counts.append(len(chunk_offset_mapping))
            
            # Move to next chunk with overlap
            start_idx += chunk_size - chunk_overlap
        
        return splits, token_counts


    

    def _split_text(self, text: str, paragraph_separator: str = None, chunk_size: int = None, chunk_overlap: int = None) -> tuple[list[str], list[int]]:
        """Split incoming text and return chunks using tokenizer."""
        final_splits: list[str] = []
        final_token_counts: list[int] = []
        
        if not text:
            return [], []
        
        if chunk_size is None:
            chunk_size = DEFAULT_CHUNK_SIZE

        if chunk_overlap is None:
            chunk_overlap = DEFAULT_CHUNK_OVERLAP

        if paragraph_separator is None:
            paragraph_separator = DEFAULT_PARAGRAPH_SEPARATOR

        # First split by paragraph separator
        paragraph_splits = self._split_text_with_regex(text, paragraph_separator)
        
        # Process each paragraph split
        for split in paragraph_splits:
            # Ensure split is a string
            if not isinstance(split, str):
                split = str(split) if split is not None else ""
            
            # Skip empty splits
            if not split:
                continue
                
            token_count = estimate_tokens_in_text(split)
            if token_count <= chunk_size:
                final_splits.append(split)
                final_token_counts.append(token_count)
            else:
                # Split further using tokenizer
                sub_splits, sub_token_counts = self._split_text_on_tokens(split, chunk_size, chunk_overlap)
                final_splits.extend(sub_splits)
                final_token_counts.extend(sub_token_counts)
        
        return final_splits, final_token_counts

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        for node in nodes_with_progress:
            splits, token_counts = self.split_text(node.get_content(metadata_mode=MetadataMode.NONE))
            node.metadata.pop("content_list", None)
            chunks = build_nodes_from_splits(splits, node, id_func=self.id_func)
            for chunk, token_count in zip(chunks, token_counts):
                chunk.metadata["token_count"] = token_count
            all_nodes.extend(chunks)
        return all_nodes

