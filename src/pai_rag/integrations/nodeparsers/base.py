"""Markdown node parser."""
from llama_index.core.bridge.pydantic import Field
from typing import Any, Iterator, List, Optional, Sequence

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable


CHUNK_CHAR_SET = set(".?!。？！\n")


class StructuredNodeParser(NodeParser):
    """Strcutured node parser.

    Will try to detect document struct according to Title information.

    Splits a document into Nodes using custom splitting logic.

    Args:
        max_chunk_size (int): max chunk size
        chunk_overlap_size (int): chunk overlap size
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    max_chunk_size: int = Field(default=500, description="Max chunk size.")
    chunk_overlap_size: int = Field(default=10, description="Chunk overlap size.")

    @classmethod
    def from_defaults(
        cls,
        max_chunk_size: int = 500,
        chunk_overlap_size: int = 10,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "StructuredNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            max_chunk_size=max_chunk_size,
            chunk_overlap_size=chunk_overlap_size,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "StructuredNodeParser"

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node)
            all_nodes.extend(nodes)

        return all_nodes

    def _extract_header_info(self, text):
        """Extract section header information from text.
           Returns section header text(str), rank(int) and header position (start, end).
        For txt text, simple return None, 0, None
        For html text like "<h1>Content</h1>", return "Content", 1, (0, 15)
        For markdown text like "## Business Justification", return "Business Justification", 2, (0, 20)
        """
        return None, 0

    def _check_plain_text(self, text, plain_text_flag):
        """Check whether current text is plain text and we don't extract structure info to these text.

        Applied to code blocks in markdown && comments in html.
        """
        return True

    def _wrap_text_line(self, text):
        """处理添加到section的文本，添加boundry。
        主要用于处理markdown断行和html跨标签的文本等等。
        """
        return text + "\n"

    def _format_section_header(self, section_headers):
        return " >> ".join([h[0] for h in section_headers])

    def _push_current_header(self, section_headers, header, rank):
        while section_headers:
            _, last_rank = section_headers[-1]
            if last_rank >= rank:
                # 上一个section是同级或者更小的标题
                section_headers.pop()
            else:
                break
        section_headers.append((header, rank))

        return

    def _cut(self, raw_section: str) -> Iterator[str]:
        if len(raw_section) <= self.max_chunk_size:
            yield raw_section
        else:
            start = 0
            while start < len(raw_section):
                end = start + self.max_chunk_size
                if end >= len(raw_section):
                    yield raw_section[start:end]
                    start = end
                    continue

                pos = end - 1

                while pos >= start + 200 and raw_section[pos] not in CHUNK_CHAR_SET:
                    pos -= 1

                yield raw_section[start : pos + 1]
                if raw_section[pos] in CHUNK_CHAR_SET:
                    start = pos + 1
                else:
                    start = pos + 1 - self.chunk_overlap_size

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """Get nodes from document."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        lines = text.split("\n")
        plain_text_flag = False
        section_headers = []  # stack for storing section headers
        current_section = ""

        sections = []
        for line in lines:
            plain_text_flag = self._check_plain_text(line, plain_text_flag)
            if not plain_text_flag:
                header, rank, pos = self._extract_header_info(text=line)
                # 识别到了header信息
                if header:
                    header_start, header_end = pos[0], pos[1]
                    current_section += self._wrap_text_line(line[:header_start])
                    if current_section.strip():
                        current_header = self._format_section_header(section_headers)
                        for section_parts in self._cut(current_section):
                            sections.append(f"{current_header}: {section_parts}")

                    self._push_current_header(
                        section_headers=section_headers, header=header, rank=rank
                    )
                    current_section = self._wrap_text_line(line[header_end:])
                else:
                    current_section += self._wrap_text_line(line)
            else:
                current_section += self._wrap_text_line(line)

        if current_section.strip():
            current_header = self._format_section_header(section_headers)
            for section_parts in self._cut(current_section):
                sections.append(f"{current_header}: {section_parts}")

        split_nodes = build_nodes_from_splits(sections, node, id_func=self.id_func)
        return split_nodes

    def _update_metadata(
        self, headers_metadata: dict, new_header: str, new_header_level: int
    ) -> dict:
        """Update the markdown headers for metadata.

        Removes all headers that are equal or less than the level
        of the newly found header
        """
        updated_headers = {}

        for i in range(1, new_header_level):
            key = f"Header_{i}"
            if key in headers_metadata:
                updated_headers[key] = headers_metadata[key]

        updated_headers[f"Header_{new_header_level}"] = new_header
        return updated_headers

    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        metadata: dict,
    ) -> TextNode:
        """Build node from single text split."""
        node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]

        if self.include_metadata:
            node.metadata = {**node.metadata, **metadata}

        return node


class MarkdownNodeParser(StructuredNodeParser):
    def _check_plain_text(self, text, plain_text_flag):
        if text.strip().startswith("```"):
            return not plain_text_flag
        return plain_text_flag

    def _extract_header_info(self, text):
        rank = 0
        while rank < len(text) and text[rank] == "#":
            rank += 1

        if rank == 0:
            return None, None, None
        else:
            return text[rank:].strip(), rank, (0, len(text))
