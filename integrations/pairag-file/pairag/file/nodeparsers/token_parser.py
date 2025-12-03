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
from loguru import logger

class TokenTextSplitter(TextSplitter):
    """Splitting text to tokens using model tokenizer."""
    chunk_size: int = Field(default=800, description="chunk size.")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size.")

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 50,
        id_func: Callable[[int, BaseNode], str] = None,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._tokenizer = get_tokenizer()



    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        logger.info(f"Splitting text into chunks with chunk size {self.chunk_size}")
        return self._split_text(text, chunk_size=self.chunk_size)

    def _split_text(self, text: str, chunk_size: int = None) -> list[str]:
        """Split incoming text and return chunks using tokenizer."""
        splits: list[str] = []
        
        if chunk_size is None:
            chunk_size = self.chunk_size
        
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
            cur_idx = min(start_idx + chunk_size, len(input_ids))
            chunk_offset_mapping = offset_mapping[start_idx:cur_idx]
            
            if chunk_offset_mapping:
                chunk_start_char = chunk_offset_mapping[0][0]
                chunk_end_char = chunk_offset_mapping[-1][1]
                # Extract text using character offsets to avoid splitting characters
                chunk_text = text[chunk_start_char:chunk_end_char]
                splits.append(chunk_text)
            
            # Move to next chunk with overlap
            start_idx += chunk_size - self.chunk_overlap
        
        return splits

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        for node in nodes_with_progress:
            splits = self.split_text(node.get_content(metadata_mode=MetadataMode.NONE))
            node.metadata.pop("content_list", None)
            all_nodes.extend(
                build_nodes_from_splits(splits, node, id_func=self.id_func)
            )
        return all_nodes

