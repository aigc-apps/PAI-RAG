from fuzzywuzzy import fuzz
from mineru.utils.enum_class import BlockType, ContentType
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text
from rag.file.utils.markdown_utils import HARD_LINE_BREAK
def extract_text_from_block(para_block, para_type):
    para_text = ""
    if para_type == BlockType.TABLE:
        for block in para_block["blocks"]:  # 1st.拼table_caption
            if block["type"] == BlockType.TABLE_CAPTION:
                para_text += merge_para_with_text(block) + HARD_LINE_BREAK
        for block in para_block["blocks"]:  # 2nd.拼table_body
            if block["type"] == BlockType.TABLE_BODY:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["type"] == ContentType.TABLE:
                            # if processed by table model
                            if span.get("latex", ""):
                                para_text += f"\n\n$\n {span['latex']}\n$\n\n"
                            elif span.get("html", ""):
                                para_text += f"\n\n{span['html']}\n\n"
    return para_text

def fuzzy_match_content(tree_node_content, para_block, para_type):
    """模糊匹配表格内容"""
    block_text = extract_text_from_block(para_block, para_type)
    ratio = fuzz.token_sort_ratio(tree_node_content, block_text) / 100.0
    return ratio
