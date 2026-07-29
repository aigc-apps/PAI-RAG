"""Markdown node parser."""
import uuid
from llama_index.core.bridge.pydantic import Field, BaseModel
from urllib.parse import urlparse
from typing import Any, Callable, Iterator, List, Optional, Sequence
from loguru import logger

from llama_index.core.node_parser.interface import NodeParser
from pairag.file.nodeparsers.sentence_parser import MySentenceSplitter
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.schema import (
    BaseNode,
    TextNode,
    NodeRelationship,
    MetadataMode,
)
from pairag.file.utils.markdown_tree_utils import (
    build_markdown_tree,
    TreeNode,
    HARD_LINE_BREAK
)
from pairag.file.utils.tokenization import get_tokenizer



class StructuredNodeParser(BaseModel):
    """Structured node parser.

    Will try to detect document struct according to Title information.

    Splits a document into Nodes using custom splitting logic.

    Args:
        chunk_size (int): chunk size
        chunk_overlap (int): chunk overlap size
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    chunk_size: int = Field(default=800, description="chunk size.")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size.")
    base_parser: NodeParser = Field(
        default=None,
        description="base parser",
    )
    id_func: Callable[[int, BaseNode], str] = Field(
        default=None,
        description="Function to generate node IDs",
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

    def _format_section_header(self, section_headers, enable_hierarchy: Optional[bool]=False) -> str:
        if not enable_hierarchy:
            return section_headers[-1].content
        else:
            return "\n".join([h.content for h in section_headers])

    def _format_tree_nodes(
        self,
        node,
        doc_node,
        ref_doc,
        nodes_list,
        title_stack,
    ) -> str:
        """
        递归处理节点，返回节点内容

        返回:
            content: 节点总内容
        """
        # 初始化当前节点的内容
        total_content = f"{node.content}\n"

        # 当前节点是叶子节点,直接返回
        if not node.children:
            return total_content

        # 非叶子节点：递归处理所有子节点
        for child in node.children:
            child_content = self._format_tree_nodes(
                child,
                doc_node,
                ref_doc,
                nodes_list,
                title_stack,
            )
            # 累加子节点内容
            total_content += f"{child_content}\n"


        return total_content

    def _create_text_node(
        self,
        chunk_content,
        doc_node,
        ref_doc,
        title_stack: Optional[List[TreeNode]] = None,
    ) -> TextNode:
        relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}
        # Each chunk must get its own metadata copy — sharing doc_node.extra_info
        # across chunks causes the last chunk's title to overwrite all others.
        chunk_metadata = dict(doc_node.extra_info)
        if title_stack:
            chunk_metadata['title'] = "\n".join([h.content for h in title_stack])
        # Use the short id_func when available (e.g. chunk_xxxxxxxx), fall back
        # to full uuid for backward compatibility.
        if self.id_func:
            chunk_id = self.id_func(0, doc_node)
        else:
            chunk_id = uuid.uuid4().hex
        text_node = TextNode(
            id_=chunk_id,
            text=chunk_content,
            embedding=doc_node.embedding,
            excluded_embed_metadata_keys=doc_node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=doc_node.excluded_llm_metadata_keys,
            metadata_separator=doc_node.metadata_separator,
            metadata_template=doc_node.metadata_template,
            text_template=doc_node.text_template,
            metadata=chunk_metadata,
            relationships=relationships,
        )

        return text_node

    def get_nodes_from_tree(
        self, root: TreeNode, doc_node: BaseNode, ref_doc: Optional[BaseNode] = None
    ) -> list[BaseNode]:
        ref_doc = ref_doc or doc_node
        nodes_list = []
        title_stack = []
        nearest_title_stack = []
        # 初始化指针
        self.traverse_tree(root, doc_node, ref_doc, nodes_list, title_stack, nearest_title_stack)

        return nodes_list.copy()

    def _split_level_nodes(self, tree_nodes: list[TreeNode]):
        tree_nodes_group = []
        tree_tokens = 0
        for tree_node in tree_nodes:
            if tree_node.category in ["image", "image_caption"]:
                # 检查添加图片后是否会超过 chunk_size
                if (
                    tree_nodes_group
                    and tree_tokens + tree_node.content_token_count <= self.chunk_size
                ):
                    # 可以添加到当前 group
                    tree_nodes_group[-1].append(tree_node)
                    tree_tokens += tree_node.content_token_count
                else:
                    # 创建新 group
                    tree_nodes_group.append([tree_node])
                    tree_tokens = tree_node.content_token_count
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

    def traverse_tree(self, tree_node, doc_node, ref_doc, nodes_list, title_stack, nearest_title_stack):
        if tree_node.category == "title":
            while title_stack and title_stack[-1].level >= tree_node.level:
                title_stack.pop()

            title_stack.append(tree_node)
            nearest_title_stack.append(tree_node)

        # 单个节点token数大于chunk_size，则需要将节点进行拆分。拆分元素里不会含有image。
        if not tree_node.children:
            if tree_node.category == "image":
                # 图片节点：如果超过 chunk_size，记录警告但仍创建（图片不能拆分）
                if tree_node.content_token_count > self.chunk_size:
                    logger.warning(
                        f"Image node token count ({tree_node.content_token_count}) exceeds chunk_size ({self.chunk_size}). "
                        f"Image cannot be split, creating chunk anyway."
                    )
                node = self._create_text_node(tree_node.content, doc_node, ref_doc)
                nodes_list.append(node)
            elif tree_node.category not in ["paragraph", "table", "html_table"]:
                # 不进行切割的节点
                if nearest_title_stack:
                    chunk_text = (
                        f"{self._format_section_header(nearest_title_stack)}\n\n{tree_node.content}"
                    )
                    nearest_title_stack.pop()
                else:
                    chunk_text = tree_node.content
                node = self._create_text_node(chunk_text, doc_node, ref_doc, title_stack)
                nodes_list.append(node)
            else:
                # 段落和表格节点：进行切割
                cut_method = self._cut
                
                for chunk_text in cut_method(tree_node.content):
                    if nearest_title_stack:
                        chunk_text = (
                            f"{self._format_section_header(nearest_title_stack)}\n\n{chunk_text}"
                        )
                        nearest_title_stack.pop()
                    node = self._create_text_node(chunk_text, doc_node, ref_doc, title_stack)
                    nodes_list.append(node)

        # 处理该节点和子节点
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
                    title_stack,
                    nearest_title_stack,
                )
            else:
                chunk_text = ""
                for child in node_group:
                    text = self._format_tree_nodes(
                        child,
                        doc_node,
                        ref_doc,
                        nodes_list,
                        title_stack,
                    )
                    chunk_text += text + "\n"
                if nearest_title_stack:
                    chunk_text = (
                        f"{self._format_section_header(nearest_title_stack)}\n\n{chunk_text}"
                    )
                    nearest_title_stack.pop()
                
                node = self._create_text_node(chunk_text, doc_node, ref_doc, title_stack)
                nodes_list.append(node)

        return


class MarkdownNodeParser(NodeParser):
    chunk_size: int = Field(default=800, description="chunk size.")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size.")
    base_parser: Any = None

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 50,
        paragraph_separator: str = "\n\n",
        base_parser: Any = None,
        id_func: Callable[[int, BaseNode], str] = None,
    ):
        super().__init__(
            id_func=id_func,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_parser = base_parser or MySentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator=paragraph_separator,
            tokenizer=get_tokenizer(),
            id_func=id_func,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        parser = StructuredNodeParser(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            base_parser=self.base_parser,
            id_func=self.id_func,
        )

        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Chunking nodes")

        all_chunks = []
        for node in nodes_with_progress:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            ast_root = build_markdown_tree(text)

            chunks = parser.get_nodes_from_tree(ast_root, node)
            all_chunks.extend(chunks)

        return all_chunks
