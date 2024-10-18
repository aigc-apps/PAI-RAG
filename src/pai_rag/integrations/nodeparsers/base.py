"""Markdown node parser."""
from llama_index.core.bridge.pydantic import Field
from typing import Any, Iterator, List, Optional, Sequence
import re

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.schema import (
    BaseNode,
    ImageNode,
    TextNode,
    NodeRelationship,
    MetadataMode,
)

CHUNK_CHAR_SET = set(".?!。？！\n")
IMAGE_URL_PATTERN = (
    r"!\[(?P<alt_text>.*?)\]\((https?://[^\s]+?[\s\w.-]*\.(jpg|jpeg|png|gif|bmp))\)"
)
ALT_REGEX_PATTERN = r"^pai_rag_image_\d{10}(_\w*)?$"


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
    enable_multimodal: bool = Field(
        default=False, description="whether use multimodal."
    )
    base_parser: NodeParser = Field(
        default=SentenceSplitter(), description="base parser"
    )

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
        return self.base_parser.split_text(raw_section)

    def _build_nodes_from_split(
        self,
        current_header: str,
        current_section: str,
        node: BaseNode,
        ref_doc: Optional[BaseNode] = None,
    ) -> List[TextNode]:
        ref_doc = ref_doc or node
        relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}
        image_urls_positions = []
        raw_section_without_image = current_section
        for match in re.finditer(IMAGE_URL_PATTERN, current_section):
            alt_text = match.group("alt_text")
            img_url = match.group(2)
            img_text = match.group(0)
            alt_text_start_pos, img_url_end_pos = match.span()
            img_info = {
                "image_url": img_url,
                "image_text": img_text,
                "image_url_start_pos": alt_text_start_pos,
                "image_url_end_pos": img_url_end_pos,
            }
            if re.match(ALT_REGEX_PATTERN, alt_text):
                image_urls_positions.append(img_info)

                raw_section_without_image = raw_section_without_image.replace(
                    img_text, ""
                )
        nodes = []
        cur_chunk_start_position = 0
        for section_parts in self._cut(raw_section_without_image):
            section_image_urls_positions = []
            node_text = f"{current_header}: {section_parts}"
            cur_chunk_end_position = cur_chunk_start_position + len(section_parts)

            for img_info in image_urls_positions:
                if (
                    cur_chunk_start_position
                    <= img_info["image_url_start_pos"]
                    <= cur_chunk_end_position
                ):
                    img_info = {
                        "image_url": img_info["image_url"],
                        "image_text": img_info["image_text"],
                        "image_url_start_pos": img_info["image_url_start_pos"]
                        - cur_chunk_start_position,
                        "image_url_end_pos": img_info["image_url_end_pos"]
                        - cur_chunk_start_position,
                    }
                    section_image_urls_positions.append(img_info)
                    cur_chunk_end_position += len(img_info["image_text"])
                    if self.enable_multimodal:
                        image_node = ImageNode(
                            embedding=node.embedding,
                            image_url=img_info["image_url"],
                            excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                            excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                            metadata_seperator=node.metadata_seperator,
                            metadata_template=node.metadata_template,
                            text_template=node.text_template,
                            metadata={
                                "image_url": img_info["image_url"],
                                **node.extra_info,
                            },
                            relationships=relationships,
                        )
                        nodes.append(image_node)
            cur_chunk_start_position = cur_chunk_end_position
            if len(section_image_urls_positions) > 0 and self.enable_multimodal:
                text_node = TextNode(
                    text=node_text,
                    embedding=node.embedding,
                    excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                    metadata_seperator=node.metadata_seperator,
                    metadata_template=node.metadata_template,
                    text_template=node.text_template,
                    metadata={
                        "image_info_list": section_image_urls_positions,
                        **node.extra_info,
                    },
                    relationships=relationships,
                )
                nodes.append(text_node)
            else:
                text_node = TextNode(
                    text=node_text,
                    embedding=node.embedding,
                    excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                    metadata_seperator=node.metadata_seperator,
                    metadata_template=node.metadata_template,
                    text_template=node.text_template,
                    meta_data=node.extra_info,
                    relationships=relationships,
                )
                nodes.append(text_node)

        return nodes

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """Get nodes from document."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        lines = text.split("\n")
        plain_text_flag = False
        section_headers = []  # stack for storing section headers
        current_section = ""

        nodes = []
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
                        nodes.extend(
                            self._build_nodes_from_split(
                                current_header, current_section, node
                            )
                        )

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
            nodes.extend(
                self._build_nodes_from_split(current_header, current_section, node)
            )
        return nodes

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
