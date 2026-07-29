"""Positional Markdown node parser."""
import uuid
from llama_index.core.bridge.pydantic import Field, BaseModel
from typing import Any, Callable, List, Sequence, Dict

from llama_index.core.node_parser.interface import NodeParser, NodeRelationship
from pairag.file.nodeparsers.sentence_parser import MySentenceSplitter
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.schema import (
    BaseNode,
    TextNode,
    MetadataMode,
)
from fastpdf4llm import ContentBlock
from pairag.file.utils.tokenization import estimate_tokens_in_text
from pairag.file.utils.tokenization import get_tokenizer
from loguru import logger
import json


# block with token count
class TokenContentBlock(ContentBlock):
    token_count: int = Field(default=0)


class Section(BaseModel):
    blocks: List[TokenContentBlock] = Field(default=[])    
    token_count: int = Field(default=0)
    level: int = -1 # -1 means the section is not a title

    def add_block(self, block: ContentBlock):
        if not self.blocks:
            self.level = block.text_level or 999
        self.blocks.append(block)
        self.token_count += block.token_count
    
    def can_merge_section(self, section: "Section", chunk_size: int) -> bool:
        if self.level > section.level or self.token_count + section.token_count > chunk_size:
            return False
        return True

    def merge_section(self, section: "Section"):
        if self.level > section.level:
            raise ValueError("The section level {section.level} is less than the current section level {self.level}.")

        self.blocks.extend(section.blocks)
        self.token_count += section.token_count

    def get_blocks(self):
        return self.blocks

    def get_token_count(self):
        return self.token_count
    
    def can_add_content(self, content: TokenContentBlock, chunk_size: int) -> bool:
        if self.token_count + content.token_count > chunk_size:
            return False
        
        if content.text_level is not None:
            return False
        
        return True
    
    def to_chunk(self, doc_node: BaseNode, metadata: Dict[str, Any], id_func: Callable = None) -> List[TextNode]:
        content = ""
        page_bbox = []
        for block in self.blocks:
            content += block.text.rstrip("\n") + "\n\n"
            page_bbox.append(
                {
                    "page_idx": block.page,
                    "bbox": block.bbox
                }
            )

        relationships = {
            NodeRelationship.SOURCE: doc_node.as_related_node_info(),
        }
        metadata["page_bbox"] = json.dumps(page_bbox)
        metadata["token_count"] = self.token_count
        if id_func:
            chunk_id = id_func(0, doc_node)
        else:
            chunk_id = uuid.uuid4().hex
        return TextNode(
            id_=chunk_id,
            text=content,
            metadata=metadata,
            relationships=relationships
        )


# Markdown node parser that keeps bbox positional information to split the nodes.
class PositionalMarkdownNodeParser(NodeParser):
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
        self.base_parser = MySentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            tokenizer=get_tokenizer(),
            id_func=id_func,
        )

    def preprocess_content_list(self, content_list: List[ContentBlock]) -> List[TokenContentBlock]:
        new_content_list = []

        i = 0
        while i < len(content_list):
            next_i = i + 1
            if content_list[i].text_level is not None and content_list[i].text_level > 0: # chunk是标题，进行标题合并
                j = i + 1
                height = content_list[i].bbox[3] - content_list[i].bbox[1]
                while j < len(content_list) and content_list[j].text_level is not None and content_list[j].text_level > 0:
                    if (abs(content_list[j].bbox[1] - content_list[i].bbox[1]) < 3
                      or (abs(content_list[j].bbox[1] - content_list[i].bbox[3]) < height / 2)):
                      logger.info(f"Merge title content {content_list[i].text} and {content_list[j].text}")
                      content_list[i].text += content_list[j].text
                      content_list[i].bbox = (
                        min(content_list[i].bbox[0], content_list[j].bbox[0]),
                        min(content_list[i].bbox[1], content_list[j].bbox[1]),
                        max(content_list[i].bbox[2], content_list[j].bbox[2]),
                        max(content_list[i].bbox[3], content_list[j].bbox[3])
                      )
                      j += 1
                    else:
                        break
                next_i = j

            token_count = estimate_tokens_in_text(content_list[i].text)
            # 当前chunk太大，需要拆分
            if token_count > self.chunk_size:
                sub_contents = self.base_parser.split_text(content_list[i].text)
                start_y = content_list[i].bbox[1]
                height = content_list[i].bbox[3] - content_list[i].bbox[1]
                for sub_content in sub_contents:
                    sub_token_count = estimate_tokens_in_text(sub_content)
                    sub_bbox = (content_list[i].bbox[0], start_y, content_list[i].bbox[2], start_y + (sub_token_count / token_count) * height)
                    sub_block = TokenContentBlock(
                        type=content_list[i].type,
                        text=sub_content,
                        text_level=content_list[i].text_level,
                        bbox=sub_bbox,
                        page=content_list[i].page,
                        token_count=sub_token_count)

                    new_content_list.append(sub_block)
                    start_y = sub_bbox[3]
            else:
                new_content_list.append(TokenContentBlock(
                        type=content_list[i].type,
                        text=content_list[i].text,
                        text_level=content_list[i].text_level,
                        bbox=content_list[i].bbox,
                        page=content_list[i].page,
                        token_count=token_count))
            i = next_i

        return new_content_list


    def split_content_list(self, content_list: List[TokenContentBlock]) -> List[Section]:
        if not content_list:
            return []
        
        stack = []

        # 我们假设每个content中的token数目是一个合理的数值
        # 当然可能出现长度很大的table之类的情形，这种情况下我们不会尝试切分table内部，我们认为表头的语义足够应付大部分情形
        # 因此当前一个比较naive且lazy的做法就是不主动切分content，这样bbox拼接会更容易一些。
        for content in content_list:            
            # 需要新建一个section的情形：
            # 1. 栈为空
            # 2. 栈顶section的token数 + 当前content的token数 >= chunk_size
            # 3. 栈顶section的level > 当前content level，说明进入新的章节，单独成section，等待后序出栈时合并
            if not stack or not stack[-1].can_add_content(content, self.chunk_size):
                while len(stack) > 1 and stack[-2].can_merge_section(stack[-1], self.chunk_size):
                    stack[-2].merge_section(stack[-1])
                    stack.pop()
                # try merge current stack top sections
                new_section = Section(blocks=[content], token_count=content.token_count)
                stack.append(new_section)
            else:
                stack[-1].add_block(content)

        while len(stack) > 1 and stack[-2].can_merge_section(stack[-1], self.chunk_size):
            stack[-2].merge_section(stack[-1])
            stack.pop()

        return stack

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Chunking nodes")

        all_chunks = []
        for node in nodes_with_progress:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            content_list = node.metadata.get("content_list", [])
            content_list = self.preprocess_content_list(content_list)

            node.metadata.pop("content_list", None)

            sections = self.split_content_list(content_list)
            for section in sections:
                chunk_metadata = node.metadata.copy()
                chunk = section.to_chunk(doc_node=node, metadata=chunk_metadata, id_func=self.id_func)
                all_chunks.append(chunk)


        return all_chunks
