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
from mistletoe.span_token import RawText, Emphasis, Strong, InlineCode, Link, Image
from mistletoe import Document
from typing import List as TList, Union


# 定义树节点的数据结构
class TreeNode:
    def __init__(self, level: int, category: str, content: str):
        self.level = level  # 层级
        self.category = category  # 所属类别
        self.content = content  # 节点内容
        self.content_token_count = len(content)  # 本节点内容和本节点所有子节点token数
        self.children: TList["TreeNode"] = []  # 子节点列表

    def add_child(self, node: "TreeNode"):
        self.children.append(node)

    def compute_total_tokens(self) -> int:
        """
        递归计算该节点及所有子节点的 token 数。
        将结果存储在 content_token_count 中。
        """
        if not self.children:
            return len(self.content)
        total = 0
        for child in self.children:
            if child.category != "image":
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

    def handle_paragraph(self, node: Paragraph):
        content = self.render_span_tokens(node.children).strip()
        if not content:
            return  # 忽略空段落

        new_node = TreeNode(
            level=self.stack[-1].level, category="paragraph", content=content
        )
        self.stack[-1].add_child(new_node)

    def handle_code_fence(self, node: CodeFence):
        language = node.language or ""
        content = node.children[0].content.rstrip() if node.children else ""
        code_content = f"{language}:\n{content}" if language else content
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
        content = f"{leader}{content}"
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
        content = node.src
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
                result += self.process_span_node(token)
        # for token in tokens:
        #     result += self.process_span_node(token)
        return result

    def process_span_node(self, node) -> str:
        if isinstance(node, RawText):
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
