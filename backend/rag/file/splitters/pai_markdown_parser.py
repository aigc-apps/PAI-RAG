"""Markdown node parser."""
import uuid
from llama_index.core.bridge.pydantic import Field, BaseModel
from urllib.parse import urlparse
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Dict

from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.schema import (
    BaseNode,
    TextNode,
    NodeRelationship,
    MetadataMode,
)
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text
from rag.file.utils.markdown_utils import (
    build_markdown_tree,
    TreeNode,
)
from rag.file.splitters.utils import fuzzy_match_content
from mineru.utils.enum_class import BlockType
from fuzzywuzzy import fuzz
from rag.file.utils.markdown_utils import HARD_LINE_BREAK
import json


# 指针类型定义
Pointer = Dict[str, int]  # {"page_idx": int, "block_idx": int}


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
        content_list,
        pointer: Pointer
    ) -> tuple:
        """
        递归处理节点，返回节点内容及该节点和所有子节点的pages_bbox集合

        返回:
            tuple: (节点总内容, 所有相关节点的pages_bbox列表, 更新后的指针)
        """
        # 初始化当前节点的内容和pages_bbox集合
        total_content = f"{node.content}\n"
        all_pages_bbox = []

        # 处理当前节点自身的pages_bbox

        self_pages_bbox, pointer = self.find_node_bbox_in_content_list(node.content, content_list, pointer)

        # 将当前节点的pages_bbox加入集合
        all_pages_bbox.extend(self_pages_bbox)


        # 当前节点是叶子节点,直接返回
        if not node.children:
            return total_content, all_pages_bbox, pointer

        # 非叶子节点：递归处理所有子节点
        for child in node.children:
            child_content, child_pages_bbox, pointer = self._format_tree_nodes(
                child,
                doc_node,
                ref_doc,
                nodes_list,
                title_stack,
                content_list,
                pointer
            )
            # 累加子节点内容
            total_content += f"{child_content}\n"
            # 合并子节点的pages_bbox
            all_pages_bbox.extend(child_pages_bbox)


        return total_content, all_pages_bbox, pointer

    def _create_text_node(
        self,
        chunk_content,
        doc_node,
        ref_doc,
        pages_bbox: Optional[List[dict]] = None,
        title_stack: Optional[List[TreeNode]] = None,
    ) -> TextNode:
        relationships = {NodeRelationship.SOURCE: ref_doc.as_related_node_info()}
        doc_node.extra_info['pages_bbox'] = json.dumps(pages_bbox, ensure_ascii=False)
        doc_node.extra_info['chapter_name'] = "\n".join([h.content for h in title_stack])
        if "content_list" in doc_node.extra_info:
            doc_node.extra_info.pop("content_list")
        text_node = TextNode(
            id_=uuid.uuid4().hex,
            text=chunk_content,
            embedding=doc_node.embedding,
            excluded_embed_metadata_keys=doc_node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=doc_node.excluded_llm_metadata_keys,
            metadata_separator=doc_node.metadata_separator,
            metadata_template=doc_node.metadata_template,
            text_template=doc_node.text_template,
            metadata=doc_node.extra_info,
            relationships=relationships,
        )

        return text_node

    def get_nodes_from_tree(
        self, root: TreeNode, doc_node: BaseNode, content_list: Optional[List] = None, ref_doc: Optional[BaseNode] = None
    ) -> list[BaseNode]:
        ref_doc = ref_doc or doc_node
        nodes_list = []
        title_stack = []
        nearest_title_stack = []
        # 初始化指针
        pointer = {"page_idx": 0, "block_idx": 0}
        pointer = self.traverse_tree(root, doc_node, ref_doc, nodes_list, title_stack, nearest_title_stack, content_list, pointer)

        return nodes_list.copy()

    def _split_level_nodes(self, tree_nodes: list[TreeNode]):
        tree_nodes_group = []
        tree_tokens = 0
        for tree_node in tree_nodes:
            if tree_node.category in ["image", "image_caption"]:
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

    def traverse_tree(self, tree_node, doc_node, ref_doc, nodes_list, title_stack, nearest_title_stack, content_list, pointer: Pointer) -> Pointer:
        if tree_node.category == "title":
            while title_stack and title_stack[-1].level >= tree_node.level:
                title_stack.pop()

            # 使用pointer和find_node_bbox_in_content_list查找title信息
            title_bbox_list, pointer = self.find_node_bbox_in_content_list(tree_node.content, content_list, pointer)

            # 如果找到了title的bbox信息，更新tree_node的属性
            if title_bbox_list:
                first_bbox = title_bbox_list[0]  # 取第一个匹配的bbox
                tree_node.page_idx = first_bbox["page_idx"]
                tree_node.bbox = first_bbox["bbox"]

            title_stack.append(tree_node)
            nearest_title_stack.append(tree_node)

        # 单个节点token数大于chunk_size，则需要将节点进行拆分。拆分元素里不会含有image。
        if not tree_node.children:
            if tree_node.category == "image":
                # 图片描述，不进行切割, 在原文中也没有bbox
                node = self._create_text_node(tree_node.content, doc_node, ref_doc)
                nodes_list.append(node)
            elif tree_node.category != "paragraph":
                # 不进行切割, 通过title的范围模糊匹配在原文中找到bbox
                pages_bbox, pointer = self.find_node_bbox_in_content_list(tree_node.content, content_list, pointer)
                if nearest_title_stack:
                    pages_bbox.append({"page_idx": nearest_title_stack[-1].page_idx, 'bbox': nearest_title_stack[-1].bbox})
                    chunk_text = (
                        f"{self._format_section_header(nearest_title_stack)}\n\n{tree_node.content}"
                    )
                    nearest_title_stack.pop()
                else:
                    chunk_text = tree_node.content
                node = self._create_text_node(chunk_text, doc_node, ref_doc, pages_bbox, title_stack)
                nodes_list.append(node)
            else:
                for chunk_text in self._cut(tree_node.content):
                    # 计算chunk的bbox
                    pages_bbox, pointer = self.find_node_bbox_in_content_list(chunk_text, content_list, pointer)
                    if nearest_title_stack:
                        pages_bbox.append({"page_idx": nearest_title_stack[-1].page_idx, 'bbox': nearest_title_stack[-1].bbox})
                        chunk_text = (
                            f"{self._format_section_header(nearest_title_stack)}\n\n{chunk_text}"
                        )
                        nearest_title_stack.pop()
                    node = self._create_text_node(chunk_text, doc_node, ref_doc, pages_bbox, title_stack)
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
                pointer = self.traverse_tree(
                    node_group[0],
                    doc_node,
                    ref_doc,
                    nodes_list,
                    title_stack,
                    nearest_title_stack,
                    content_list,
                    pointer
                )
            else:
                chunk_text = ""
                all_pages_bbox = []
                for child in node_group:
                    text, pages_bbox, pointer = self._format_tree_nodes(
                        child,
                        doc_node,
                        ref_doc,
                        nodes_list,
                        title_stack,
                        content_list,
                        pointer
                    )
                    chunk_text += text + "\n"
                    all_pages_bbox.extend(pages_bbox)
                if nearest_title_stack:
                    all_pages_bbox.append({"page_idx": nearest_title_stack[-1].page_idx, 'bbox': nearest_title_stack[-1].bbox})
                    chunk_text = (
                        f"{self._format_section_header(nearest_title_stack)}\n\n{chunk_text}"
                    )
                    nearest_title_stack.pop()
                node = self._create_text_node(chunk_text, doc_node, ref_doc, all_pages_bbox, title_stack)
                nodes_list.append(node)

        return pointer

    def find_node_bbox_in_content_list(self, node_content, content_list, pointer: Pointer) -> Tuple[List[dict], Pointer]:
        if not content_list:
            return [], pointer

        pages_bbox = []
        start_page_idx = pointer["page_idx"]
        start_block_idx = pointer["block_idx"]


        # 从指针位置开始查找
        for i in range(start_page_idx, len(content_list)):
            content = content_list[i]
            para_blocks = content.get('preproc_blocks', [])
            block_start = start_block_idx if i == start_page_idx else 0

            for j in range(block_start, len(para_blocks)):
                para_block = para_blocks[j]
                if para_block.get('type')  == BlockType.TABLE:
                    ratio = fuzzy_match_content(node_content, para_block, para_block["type"])
                    if ratio > 0.7:
                        pages_bbox.append({
                            "page_idx": content.get("page_idx", -1),
                            'bbox': para_block['bbox']
                        })
                        # 更新指针到下一个位置
                        pointer = {"page_idx": i, "block_idx": j + 1}
                        return pages_bbox, pointer
                elif para_block.get('type') == BlockType.IMAGE:
                    para_text = ""
                    for block in para_block["blocks"]:
                        if block["type"] == BlockType.IMAGE_CAPTION:
                            para_text += merge_para_with_text(block) + HARD_LINE_BREAK
                        elif block["type"] == BlockType.IMAGE_FOOTNOTE:
                            para_text += merge_para_with_text(block) + HARD_LINE_BREAK
                    ratio = fuzz.token_sort_ratio(node_content, para_text.strip()) / 100.0
                    if ratio > 0.9:
                        para_text_bbox = para_block.get('bbox')
                        pages_bbox.append({
                            "page_idx": content.get("page_idx", -1),
                            'bbox': para_text_bbox
                        })
                        # 更新指针到下一个位置
                        pointer = {"page_idx": i, "block_idx": j + 1}
                        return pages_bbox, pointer
                elif para_block.get('type') in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX, BlockType.TITLE, BlockType.INTERLINE_EQUATION]:
                    para_text = merge_para_with_text(para_block)
                    ratio = fuzz.token_sort_ratio(node_content, para_text.strip()) / 100.0
                    if ratio > 0.9:
                        para_text_bbox = para_block.get('bbox')
                        pages_bbox.append({
                            "page_idx": content.get("page_idx", -1),
                            'bbox': para_text_bbox
                        })
                        # 更新指针到下一个位置
                        pointer = {"page_idx": i, "block_idx": j + 1}
                        return pages_bbox, pointer

        return pages_bbox, pointer


class MarkdownNodeParser(NodeParser):
    chunk_size: int = Field(default=800, description="chunk size.")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size.")
    base_parser: Any = None

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 50,
        id_func: Callable[[int, BaseNode], str] = None,
    ):
        super().__init__(
            id_func=id_func,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
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
        )

        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Chunking nodes")

        all_chunks = []
        for node in nodes_with_progress:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            content_list = json.loads(node.metadata.get("content_list", "[]"))
            ast_root = build_markdown_tree(text)

            chunks = parser.get_nodes_from_tree(ast_root, node, content_list)
            all_chunks.extend(chunks)


        return all_chunks
