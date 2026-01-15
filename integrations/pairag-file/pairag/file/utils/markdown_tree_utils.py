from typing import List as TList, Optional, Union
from llama_index.core.bridge.pydantic import Field, BaseModel
from mistletoe.block_token import (
    Heading,
    Paragraph,
    CodeFence,
    List,
    ListItem,
    Table,
    Quote,
    HTMLBlock,
)
from mistletoe.span_token import RawText, Emphasis, Strong, InlineCode, Link, Image, LineBreak
from mistletoe import Document
from pairag.file.utils.image_utils import markdown_image_text_to_chunk
from pairag.file.utils.tokenization import estimate_tokens_in_text

START_HTML_TAG = "<html><body><table>"
END_HTML_TAG = "</table></body></html>"
START_SIMPLE_TABLE_TAG = "<table>"
END_SIMPLE_TABLE_TAG = "</table>"
HARD_LINE_BREAK = "  \n"


class PaiTable(BaseModel):
    data: TList[TList[str]] = Field(description="The table data.", default=[])
    row_headers_index: Optional[TList[int]] = Field(
        description="The table row headers index.", default=None
    )
    column_headers_index: Optional[TList[int]] = Field(
        description="The table column headers index.", default=None
    )

    def get_row_numbers(self):
        return len(self.data)

    def get_col_numbers(self):
        return len(self.data[0])

    def get_row_headers(self):
        if not self.row_headers_index or len(self.row_headers_index) == 0:
            return []
        return [self.data[row] for row in self.row_headers_index]

    def get_rows(self):
        if self.row_headers_index and len(self.row_headers_index) > 0:
            data_row_start_index = max(self.row_headers_index) + 1
        else:
            data_row_start_index = 0
        return self.data[data_row_start_index:]

    def get_column_headers(self):
        if not self.column_headers_index or len(self.column_headers_index) == 0:
            return []
        # return [self.data[:, col] for col in self.column_headers_index]
        return [[row[col] for row in self.data] for col in self.column_headers_index]

    def get_columns(self):
        if self.column_headers_index and len(self.column_headers_index) > 0:
            data_col_start_index = max(self.column_headers_index) + 1
        else:
            data_col_start_index = 0
        return [
            [row[col] for row in self.data]
            for col in range(data_col_start_index, self.get_col_numbers())
        ]


def convert_table_to_markdown(table: PaiTable, total_cols: int) -> str:
    markdown = []
    if len(table.get_column_headers()) > 0:
        headers = table.get_column_headers()
        rows = table.get_columns()
        total_cols = table.get_row_numbers()
    else:
        headers = table.get_row_headers()
        rows = table.get_rows()
    if headers:
        for header in headers:
            markdown.append("| " + " | ".join(header) + " |")
        markdown.append("| " + " | ".join(["---"] * total_cols) + " |")
    if rows:
        for row in rows:
            markdown.append("| " + " | ".join(row) + " |")
    return "\n".join(markdown)


def is_horizontal_table(table: TList[TList]) -> bool:
    # if the table is empty or the first (header) of table is empty, it's not a horizontal table
    if not table or not table[0]:
        return False

    vertical_value_any_count = 0
    horizontal_value_any_count = 0
    vertical_value_all_count = 0
    horizontal_value_all_count = 0

    """If it is a horizontal table, the probability that each row contains at least one number is higher than the probability that each column contains at least one number.
    If it is a horizontal table with headers, the number of rows that are entirely composed of numbers will be greater than the number of columns that are entirely composed of numbers.
    """

    for row in table:
        if any(isinstance(item, (int, float)) for item in row):
            horizontal_value_any_count += 1
        if all(isinstance(item, (int, float)) for item in row):
            horizontal_value_all_count += 1

    for col in zip(*table):
        if any(isinstance(item, (int, float)) for item in col):
            vertical_value_any_count += 1
        if all(isinstance(item, (int, float)) for item in col):
            vertical_value_all_count += 1

    return (
        horizontal_value_any_count >= vertical_value_any_count
        or horizontal_value_all_count > 0 >= vertical_value_all_count
    )



# 定义树节点的数据结构
class TreeNode:
    def __init__(self, level: int, category: str, content: str):
        self.level = level  # 层级
        self.category = category  # 所属类别
        self.content = content  # 节点内容
        self.content_token_count = estimate_tokens_in_text(content)  # 本节点内容和本节点所有子节点token数
        self.children: TList["TreeNode"] = []  # 子节点列表
        self.page_idx = None # 页码
        self.bbox = None # 位置坐标

    def add_child(self, node: "TreeNode"):
        self.children.append(node)

    def compute_total_tokens(self) -> int:
        """
        递归计算该节点及所有子节点的 token 数。
        将结果存储在 content_token_count 中。
        """
        if not self.children:
            return estimate_tokens_in_text(self.content)
        total = 0
        for child in self.children:
            total += child.compute_total_tokens()
        self.content_token_count += total
        return self.content_token_count

    def to_dict(self):
        return {
            "level": self.level,
            "category": self.category,
            "content": self.content,
            "content_token_count": self.content_token_count,
            "children": [child.to_dict() for child in self.children],
        }
    
    def print_tree(self, indent: int = 0, max_content_length: int = 100, show_content: bool = True):
        """
        打印树结构
        
        Args:
            indent: 缩进级别
            max_content_length: 内容最大显示长度
            show_content: 是否显示内容
        """
        prefix = "  " * indent
        content_preview = self.content[:max_content_length] + "..." if len(self.content) > max_content_length else self.content
        content_preview = content_preview.replace('\n', '\\n')
        
        if show_content:
            print(f"{prefix}[{self.category}] (level={self.level}, tokens={self.content_token_count})")
            print(f"{prefix}  Content: {content_preview}")
        else:
            print(f"{prefix}[{self.category}] (level={self.level}, tokens={self.content_token_count}, children={len(self.children)})")
        
        for child in self.children:
            child.print_tree(indent + 1, max_content_length, show_content)
    
    def __str__(self):
        """字符串表示"""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"TreeNode(category={self.category}, level={self.level}, tokens={self.content_token_count}, children={len(self.children)})"


# 自定义AST遍历器，用于构建树结构
class ASTTreeBuilder:
    def __init__(self):
        self.root = TreeNode(level=0, category="root", content="")
        self.stack = [self.root]  # 使用堆栈来跟踪当前的父节点

    def build_tree(self, node):
        """
        递归地遍历Mistletoe的AST节点，并构建自定义的树结构。
        """
        if isinstance(node, Heading):
            self.handle_heading(node)
        elif isinstance(node, Paragraph):
            self.handle_paragraph(node)
        elif isinstance(node, CodeFence):
            self.handle_code_fence(node)
        elif isinstance(node, List):
            self.handle_list(node)
        elif isinstance(node, Table):
            self.handle_table(node)
        elif isinstance(node, Quote):
            self.handle_quote(node)
        elif isinstance(node, Image):
            self.handle_image(node)
        elif isinstance(node, HTMLBlock):
            self.handle_html(node)
        elif isinstance(node, ListItem):
            self.handle_list_item(node)
        else:
            # 其他类型的节点可以根据需要处理
            pass

    def handle_heading(self, node: Heading):
        content = self.render_span_tokens(node.children).strip()
        if not content:
            return  # 忽略空标题

        new_node = TreeNode(level=node.level, category="title", content=content)

        # 调整堆栈以找到正确的父节点
        while (
            self.stack
            and self.stack[-1].level >= new_node.level
            and self.stack[-1].category != "root"
        ):
            self.stack.pop()

        # 将新节点添加为当前父节点的子节点
        self.stack[-1].add_child(new_node)
        self.stack.append(new_node)  # 将新节点压入堆栈

    def _remove_html_table_tags(self, content: str) -> str:
        """
        移除HTML标签，保留表格内容。
        """
        if START_HTML_TAG in content and END_HTML_TAG in content:
            content = content.replace(START_HTML_TAG, START_SIMPLE_TABLE_TAG).replace(END_HTML_TAG, END_SIMPLE_TABLE_TAG)
        return content

    def handle_paragraph(self, node: Paragraph):
        content = self.render_span_tokens(node.children).strip()
        if not content:
            return  # 忽略空段落

        if (content.startswith(START_HTML_TAG) and content.endswith(END_HTML_TAG)) or (content.startswith(START_SIMPLE_TABLE_TAG) and content.endswith(END_SIMPLE_TABLE_TAG)):
            new_node = TreeNode(
                level=self.stack[-1].level + 1, category="html_table", content=self._remove_html_table_tags(content)
            )
        elif content.startswith("<img") and (content.endswith(">") or content.endswith("/>")):
            new_node = TreeNode(
                level=self.stack[-1].level + 1, category="image_caption", content=content
            )
        else:
            new_node = TreeNode(
                level=self.stack[-1].level + 1, category="paragraph", content=self._remove_html_table_tags(content)
            )
        self.stack[-1].add_child(new_node)

    def handle_code_fence(self, node: CodeFence):
        language = node.language or ""
        content = node.children[0].content.rstrip() if node.children else ""
        code_content = f"\n```{language}\n{content}\n```\n"
        new_node = TreeNode(
            level=self.stack[-1].level + 1, category="code", content=code_content
        )
        self.stack[-1].add_child(new_node)

    def handle_list(self, node: List):
        category = "ordered" if node.start is not None else "unordered"
        new_node = TreeNode(
            level=self.stack[-1].level + 1, category=category, content=""
        )
        self.stack[-1].add_child(new_node)
        self.stack.append(new_node)  # 列表节点入栈

        for child in node.children:
            self.build_tree(child)

        if self.stack and self.stack[-1].category != "root":
            self.stack.pop()  # 列表节点出栈

    def handle_list_item(self, node: ListItem):
        # 列表项可能包含段落、子列表等
        content = self.render_span_tokens(node.children).strip()
        if node.leader:
            # 保留原始序号
            leader = node.leader
        else:
            leader = ""
        content = f"{leader}{content}  \n"
        new_node = TreeNode(
            level=self.stack[-1].level + 1, category="list_item", content=content
        )
        self.stack[-1].add_child(new_node)

        # 将列表项节点压入堆栈，以处理其子节点
        self.stack.append(new_node)

        # 处理列表项中的子节点（例如嵌套列表）
        for child in node.children:
            self.build_tree(child)

        if self.stack and self.stack[-1].category != "root":
            self.stack.pop()  # 列表节点出栈

    def handle_table(self, node: Table):
        table_content = self.reconstruct_table(node)
        if not table_content:
            return  # 忽略空表格

        new_node = TreeNode(
            level=self.stack[-1].level + 1, category="table", content=table_content
        )
        self.stack[-1].add_child(new_node)

    def handle_quote(self, node: Quote):
        # 引用可能包含段落、列表等
        content = self.render_span_tokens(node.children).strip()
        if not content:
            return  # 忽略空引用

        new_node = TreeNode(
            level=self.stack[-1].level + 1, category="quote", content=content
        )
        self.stack[-1].add_child(new_node)

    def handle_image(self, node: Image):
        image_url = node.src
        alt_text = self.render_span_tokens(node.children)
        if alt_text == "image.png":
            alt_text = None
        content = markdown_image_text_to_chunk(image_url, alt_text)
        new_node = TreeNode(
            level=self.stack[-1].level + 1, category="image", content=content
        )
        self.stack[-1].add_child(new_node)

    def handle_html(self, node: HTMLBlock):
        content = node.content.strip()
        if not content:
            return  # 忽略空的HTML块

        new_node = TreeNode(
            level=self.stack[-1].level + 1, category="html", content=content
        )
        self.stack[-1].add_child(new_node)

    def reconstruct_table(self, node: Table) -> str:
        """
        将Table节点重新构造为Markdown表格格式的字符串。
        """
        # 获取表头
        header_line = ""
        separator_line = ""
        if node.header:
            header_cells = [
                self.render_span_tokens(cell.children).strip()
                for cell in node.header.children
            ]
            header_line = "| " + " | ".join(header_cells) + " |"
            separator_line = "| " + " | ".join(["---"] * len(header_cells)) + " |"

        # 获取表格主体
        body_lines = []
        for row in node.children:
            row_cells = [
                self.render_span_tokens(cell.children).strip() for cell in row.children
            ]
            body_lines.append("| " + " | ".join(row_cells) + " |")

        # 组合成完整的表格
        if header_line:
            table_markdown = "\n".join([header_line, separator_line] + body_lines)
        else:
            table_markdown = "\n".join(body_lines)

        return table_markdown

    def render_span_tokens(
        self, tokens: TList[Union[RawText, Emphasis, Strong, InlineCode, Link, Image]]
    ) -> str:
        """
        渲染行内的 span tokens（如文本、加粗、斜体、链接、图片等）为字符串。
        """
        result = ""
        for token in tokens:
            if isinstance(token, Image):
                self.handle_image(token)
            else:
                processed = self.process_span_node(token)
                if processed is not None:
                    result += processed
        # for token in tokens:
        #     result += self.process_span_node(token)
        return result

    def process_span_node(self, node) -> str:
        if isinstance(node, LineBreak):  # 处理硬换行
            return HARD_LINE_BREAK
        elif isinstance(node, RawText):
            return node.content
        elif isinstance(node, Emphasis):
            content = self.render_span_tokens(node.children)
            return f"*{content}*"
        elif isinstance(node, Strong):
            content = self.render_span_tokens(node.children)
            return f"**{content}**"
        elif isinstance(node, InlineCode):
            content = node.children[0].content if node.children else ""
            return f"`{content}`"
        elif isinstance(node, Link):
            content = self.render_span_tokens(node.children)
            return f"[{content}]({node.target})"
        elif isinstance(node, Image):
            return node.src
        else:
            # 处理其他行内元素，如删除线（Strikethrough）
            return ""

    def get_tree(self) -> TreeNode:
        return self.root


def build_markdown_tree(markdown_text: str) -> TreeNode:
    doc = Document(markdown_text)
    builder = ASTTreeBuilder()

    for node in doc.children:
        builder.build_tree(node)
    ast_root = builder.get_tree()
    ast_root.compute_total_tokens()
    return ast_root
