"""
MinerU utils: extract content block from mineru raw block

Why not use content_list directly?
The bbox is not accurate in content_list, so we need to extract it from raw block.
https://github.com/opendatalab/MinerU/issues/3927
"""

import os
import re
from loguru import logger
from typing import Optional, Tuple
from markdownify import markdownify
from fastpdf4llm import ContentBlock
from pairag.file.store.base import BaseFileStore
from pairag.file.utils.image_caption_tool import ImageCaptionTool
from pairag.file.utils.image_utils import get_image_from_url, markdown_image_text_to_chunk


default_delimiters = {
    'display': {'left': '$$', 'right': '$$'},
    'inline': {'left': '$', 'right': '$'}
}

display_left_delimiter = default_delimiters['display']['left']
display_right_delimiter = default_delimiters['display']['right']
inline_left_delimiter = default_delimiters['inline']['left']
inline_right_delimiter = default_delimiters['inline']['right']



class BlockType:
    IMAGE = 'image'
    TABLE = 'table'
    IMAGE_BODY = 'image_body'
    TABLE_BODY = 'table_body'
    IMAGE_CAPTION = 'image_caption'
    TABLE_CAPTION = 'table_caption'
    IMAGE_FOOTNOTE = 'image_footnote'
    TABLE_FOOTNOTE = 'table_footnote'
    TEXT = 'text'
    TITLE = 'title'
    INTERLINE_EQUATION = 'interline_equation'
    LIST = 'list'
    INDEX = 'index'
    DISCARDED = 'discarded'

    # Added in vlm 2.5
    CODE = "code"
    CODE_BODY = "code_body"
    CODE_CAPTION = "code_caption"
    ALGORITHM = "algorithm"
    REF_TEXT = "ref_text"
    PHONETIC = "phonetic"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    ASIDE_TEXT = "aside_text"
    PAGE_FOOTNOTE = "page_footnote"


class ContentType:
    IMAGE = 'image'
    TABLE = 'table'
    TEXT = 'text'
    INTERLINE_EQUATION = 'interline_equation'
    INLINE_EQUATION = 'inline_equation'
    EQUATION = 'equation'
    CODE = 'code'


def merge_para_with_text(para_block: dict) -> Tuple[str, Tuple[float, float, float, float]]:
    para_text = ''
    bbox = None
    for line in para_block['lines']:
        if not bbox:
            bbox = line['bbox']
        else:
            bbox = (
                min(bbox[0], line['bbox'][0]),
                min(bbox[1], line['bbox'][1]),
                max(bbox[2], line['bbox'][2]),
                max(bbox[3], line['bbox'][3])
            )
        for j, span in enumerate(line['spans']):
            span_type = span['type']
            content = ''
            if span_type == ContentType.TEXT:
                content = span['content'] + " \n"
            elif span_type == ContentType.INLINE_EQUATION:
                content = f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
            elif span_type == ContentType.INTERLINE_EQUATION:
                content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"
            if content:
                if span_type in [ContentType.TEXT, ContentType.INLINE_EQUATION]:
                    if j == len(line['spans']) - 1:
                        para_text += content
                    else:
                        para_text += f'{content} '
                elif span_type == ContentType.INTERLINE_EQUATION:
                    para_text += content
    return para_text, bbox


def get_title_level(block):
    title_level = block.get('level', 1)
    if title_level > 4:
        title_level = 4
    elif title_level < 1:
        title_level = 0
    return title_level

def make_content_block(
    raw_block: dict,
    page_idx: int,
    image_store: BaseFileStore = None,
    image_local_dir: str = None,
    save_image_template: str = None,
    image_caption_tool: ImageCaptionTool = None,
    tenant_id: str = None,
    ) -> ContentBlock:
    if not raw_block.get("lines") and not raw_block.get("blocks"):
        return None

    block_text = ""
    block_bbox = None
    block_content_type = "text"
    block_text_level = None

    block_type = raw_block.get("type", "text")
    if block_type in [BlockType.TEXT, BlockType.INTERLINE_EQUATION, BlockType.PHONETIC, BlockType.REF_TEXT, BlockType.LIST]:
        block_text, block_bbox = merge_para_with_text(raw_block)
    elif block_type == BlockType.TITLE:
        block_text, block_bbox = merge_para_with_text(raw_block)
        block_text_level = get_title_level(raw_block)
    elif block_type == BlockType.IMAGE:
        if not image_store:
            logger.info("Image store is not set, skip image block.")
            return None
        block_content_type = "image"
        image_body = ""
        image_caption = ""
        image_footnote = ""
        for block in raw_block['blocks']:
            if block['type'] == BlockType.IMAGE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.IMAGE:
                            block_bbox = span['bbox']
                            if span.get('image_path', ''):
                                image_path = os.path.join(image_local_dir, span['image_path'])

                                image_file, image_name = get_image_from_url(image_path)
                                if image_name:
                                    save_image_name = save_image_template.format(image_name)

                                    try:
                                        upload_result = image_store.write(file=image_file, file_name=image_name, file_path=save_image_name, tenant_id=tenant_id)
                                        image_file.seek(0)
                                        image_alt_text = image_caption_tool.extract_image(image_file.read())
                                        if image_alt_text:
                                            cleaned_alt = re.sub(r'\n', ' ', image_alt_text).replace('\r', '').strip()
        
                                            image_body = markdown_image_text_to_chunk(upload_result.file_path, cleaned_alt)
                                            logger.info(
                                                f"Successfully saved image {upload_result.file_path} from URL: {image_path}"
                                            )
                                    except Exception as ex:
                                        logger.exception(
                                            f"Failed to save image from URL: {image_path}. Error: {ex}"
                                        )

            if block['type'] == BlockType.IMAGE_CAPTION:
                image_caption,_ = merge_para_with_text(block)
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                image_footnote, _ = merge_para_with_text(block)
        if image_caption:
            block_text += f"##### {image_caption.rstrip()}\n\n"
        if image_body:
            block_text += image_body.rstrip() + "\n\n"
        if image_footnote:
            block_text += f"{image_footnote.rstrip()}\n\n"
    elif block_type == BlockType.TABLE:
        block_content_type = "table"
        table_caption = ""
        table_body = ""
        table_foot = ""
        for block in raw_block['blocks']:
            if block['type'] == BlockType.TABLE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.TABLE:
                            if span.get('html', ''):
                                table_body = markdownify(span["html"])
                block_bbox = block['bbox']

            if block['type'] == BlockType.TABLE_CAPTION:
                table_caption, _ = merge_para_with_text(block) # caption_bbox is not accurate
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                table_foot, _ = merge_para_with_text(block) # foot_bbox is not accurate
        
        if table_caption:
            block_text += f"##### {table_caption.rstrip()}\n\n"
        if table_body:
            block_text += table_body.rstrip() + "\n\n"
        if table_foot:
            block_text += f"{table_foot.rstrip()}\n\n"
    
    elif block_type == BlockType.CODE:
        code_body = ""
        code_caption = ""
        guess_lang = ""
        for block in raw_block['blocks']:
            if block['type'] == BlockType.CODE_BODY:
                code_body = merge_para_with_text(block)
                block_bbox = block['bbox']
                if raw_block["sub_type"] == BlockType.CODE:
                    guess_lang = raw_block["guess_lang"]
            if block['type'] == BlockType.CODE_CAPTION:
                code_caption, _ = merge_para_with_text(block)
        if code_caption:
            block_text += f"##### {code_caption}\n\n"
        block_text = f"```{guess_lang}\n{code_body}\n```"

    block = ContentBlock(
        type=block_content_type,
        text=block_text,
        bbox=block_bbox or raw_block['bbox'],
        page=page_idx,
        text_level=block_text_level,
    )
    return block
    
