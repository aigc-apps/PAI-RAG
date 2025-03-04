"""Markdown node parser."""
from llama_index.core.bridge.pydantic import Field, BaseModel
from urllib.parse import urlparse
from typing import Any, Iterator, List, Optional, Sequence

from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.schema import (
    BaseNode,
    TextNode,
    NodeRelationship,
    MetadataMode,
)
from pai_rag.integrations.nodeparsers.utils.pai_markdown_tree import (
    build_markdown_tree,
    TreeNode,
)


class ImageInfo(BaseModel):
    image_url: str = Field(description="Image url.")
    image_text: Optional[str] = Field(description="Image text.", default=None)
    image_url_start_pos: Optional[int] = Field(
        description="Image start position.", default=None
    )
    image_url_end_pos: Optional[int] = Field(
        description="Image end position.", default=None
    )


class StructuredNodeParser(BaseModel):
    """Strcutured node parser.

    Will try to detect document struct according to Title information.

    Splits a document into Nodes using custom splitting logic.

    Args:
        chunk_size (int): chunk size
        chunk_overlap_size (int): chunk overlap size
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    chunk_size: int = Field(default=500, description="chunk size.")
    chunk_overlap_size: int = Field(default=10, description="Chunk overlap size.")
    image_caption_tool: Any = Field(
        default=None, description="use multimodal llm for image captioning."
    )
    base_parser: NodeParser = Field(
        default=SentenceSplitter(chunk_size=500, chunk_overlap=10),
        description="base parser",
    )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "StructuredNodeParser"

    def normalize_url(self, url: str) -> str:
        """自动补全缺失协议头的URL"""
        parsed = urlparse(url)
        if not parsed.scheme:
            # 默认使用HTTPS协议
            return f"https:{url}" if url.startswith("//") else f"https://{url}"
        return url

    def _cut(self, raw_section: str) -> Iterator[str]:
        # 可能存在单个node 字符数大于chunk_size，此时需要将node进行拆分。拆分元素里不会含有image。
        return self.base_parser.split_text(raw_section)

    def _format_section_header(self, section_headers) -> str:
        return "\n".join([h.content for h in section_headers])

    def _format_tree_nodes(
        self, node, doc_node, ref_doc, nodes_list, chunk_images_list
    ) -> str:
        if (
            node.category == "image"
            and self.image_caption_tool
            and node.content
            and node.content != "None"
        ):
            image_url = self.normalize_url(node.content)
            image_text = self.image_caption_tool.extract_url(image_url)
            """
            relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}
            new_node = TextNode(
                text=image_text,
                metadata={
                    "image_url": image_url,
                    **doc_node.extra_info,
                },
                excluded_embed_metadata_keys=doc_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=doc_node.excluded_llm_metadata_keys,
                relationships=relationships,
            )
            nodes_list.append(new_node)
            """
            return f"{image_text}\n图片链接: {image_url}\n"
        if not node.children:
            return node.content
        return node.content + "\n".join(
            [
                self._format_tree_nodes(
                    child, doc_node, ref_doc, nodes_list, chunk_images_list
                )
                for child in node.children
            ]
        )

    def _create_text_node(
        self, chunk_content, doc_node, ref_doc, nodes_list, chunk_images_list
    ) -> TextNode:
        relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}
        if len(chunk_images_list) > 0 and self.enable_multimodal:
            text_node = TextNode(
                text=chunk_content,
                embedding=doc_node.embedding,
                excluded_embed_metadata_keys=doc_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=doc_node.excluded_llm_metadata_keys,
                metadata_separator=doc_node.metadata_separator,
                metadata_template=doc_node.metadata_template,
                text_template=doc_node.text_template,
                metadata={
                    "image_info_list": chunk_images_list.copy(),
                    **doc_node.extra_info,
                },
                relationships=relationships,
            )
        else:
            text_node = TextNode(
                text=chunk_content,
                embedding=doc_node.embedding,
                excluded_embed_metadata_keys=doc_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=doc_node.excluded_llm_metadata_keys,
                metadata_separator=doc_node.metadata_separator,
                metadata_template=doc_node.metadata_template,
                text_template=doc_node.text_template,
                meta_data=doc_node.extra_info,
                relationships=relationships,
            )

        return text_node

    def get_nodes_from_tree(
        self, root: TreeNode, doc_node: BaseNode, ref_doc: Optional[BaseNode] = None
    ) -> list[BaseNode]:
        ref_doc = ref_doc or doc_node
        nodes_list = []
        chunk_images_list = []
        title_stack = []
        # 判断是否可以将整个树节点作为一个chunk
        if root.content_token_count <= self.chunk_size:
            new_chunk_text = self._format_tree_nodes(
                root, doc_node, ref_doc, nodes_list, chunk_images_list
            )
            # 避免插入内容为空的节点
            if len(new_chunk_text) > 0:
                node = self._create_text_node(
                    new_chunk_text, doc_node, ref_doc, nodes_list, chunk_images_list
                )
                nodes_list.append(node)
                chunk_images_list.clear()
        else:
            self.traverse_tree(
                root, doc_node, ref_doc, nodes_list, chunk_images_list, title_stack
            )

        return nodes_list.copy()

    def _split_level_nodes(self, tree_nodes: list[TreeNode]):
        tree_nodes_group = []
        tree_tokens = 0
        for tree_node in tree_nodes:
            if tree_node.category == "image" and self.enable_multimodal:
                if tree_nodes_group:
                    tree_nodes_group[-1].append(tree_node)
                else:
                    tree_nodes_group.append([tree_node])
            elif tree_node.content_token_count > self.chunk_size:
                if tree_nodes_group and len(tree_nodes_group[-1]) == 0:
                    tree_nodes_group[-1].append(tree_node)
                else:
                    tree_nodes_group.append([tree_node])
                tree_nodes_group.append([])
                tree_tokens = 0
            elif (
                tree_nodes_group
                and tree_tokens + tree_node.content_token_count <= self.chunk_size
            ):
                tree_nodes_group[-1].append(tree_node)
                tree_tokens += tree_node.content_token_count
            else:
                tree_nodes_group.append([tree_node])
                tree_tokens = tree_node.content_token_count
        return tree_nodes_group

    def traverse_tree(
        self, tree_node, doc_node, ref_doc, nodes_list, chunk_images_list, title_stack
    ):
        if tree_node.category == "title":
            while title_stack and title_stack[-1].level >= tree_node.level:
                title_stack.pop()
            title_stack.append(tree_node)

        # 单个节点token数大于chunk_size，则需要将节点进行拆分。拆分元素里不会含有image。
        if not tree_node.children:
            for chunk_text in self._cut(tree_node.content):
                if title_stack:
                    new_chunk_text = (
                        f"{self._format_section_header(title_stack)} : {chunk_text}"
                    )
                else:
                    new_chunk_text = chunk_text
                node = self._create_text_node(
                    new_chunk_text, doc_node, ref_doc, nodes_list, chunk_images_list
                )
                nodes_list.append(node)
                chunk_images_list.clear()

        nodes_groups = self._split_level_nodes(tree_node.children)
        for node_group in nodes_groups:
            if len(node_group) == 0:
                continue
            # 一个group里只有一个节点，且该节点的size数超过chunk_size
            if (
                len(node_group) == 1
                and node_group[0].content_token_count >= self.chunk_size
            ):
                self.traverse_tree(
                    node_group[0],
                    doc_node,
                    ref_doc,
                    nodes_list,
                    chunk_images_list,
                    title_stack,
                )
            else:
                chunk_text = ""
                for child in node_group:
                    if (
                        child.category == "image"
                        and self.image_caption_tool
                        and child.content
                        and child.content != "None"
                    ):
                        image_url = self.normalize_url(child.content)
                        image_text = self.image_caption_tool.extract_url(image_url)
                        """
                        relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}
                        new_node = TextNode(
                            text=image_text,
                            metadata={
                                "image_url": image_url,
                                **doc_node.extra_info,
                            },
                            excluded_embed_metadata_keys=doc_node.excluded_embed_metadata_keys,
                            excluded_llm_metadata_keys=doc_node.excluded_llm_metadata_keys,
                            relationships=relationships,
                        )
                        nodes_list.append(new_node)
                        """
                        chunk_text += f"\n{image_text}\n图片链接: {image_url}\n"
                    else:
                        chunk_text += "\n" + self._format_tree_nodes(
                            child, doc_node, ref_doc, nodes_list, chunk_images_list
                        )
                if title_stack:
                    new_chunk_text = (
                        f"{self._format_section_header(title_stack)} : {chunk_text}"
                    )
                else:
                    new_chunk_text = chunk_text
                node = self._create_text_node(
                    new_chunk_text, doc_node, ref_doc, nodes_list, chunk_images_list
                )
                nodes_list.append(node)
                chunk_images_list.clear()


class MarkdownNodeParser(NodeParser):
    chunk_size: int = Field(default=500, description="chunk size.")
    chunk_overlap_size: int = Field(default=10, description="Chunk overlap size.")
    image_caption_tool: Any = Field(
        default=None, description="Image caption with multimodal model."
    )
    base_parser: NodeParser = Field(
        default=SentenceSplitter(chunk_size=500, chunk_overlap=10),
        description="base parser",
    )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        parser = StructuredNodeParser(
            chunk_size=self.chunk_size,
            chunk_overlap_size=self.chunk_overlap_size,
            image_caption_tool=self.image_caption_tool,
            base_parser=self.base_parser,
        )

        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            ast_root = build_markdown_tree(text)
            nodes = parser.get_nodes_from_tree(ast_root, node)
            all_nodes.extend(nodes)
        return all_nodes
