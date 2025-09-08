from collections import defaultdict
from io import BytesIO
from operator import itemgetter
import os
import re
from rag.file.models.file_item import FileItem
from rag.file.readers.base import BaseReader
from rag.file.store.base import BaseFileStore
from rag.file.store.oss_store import OssFileStore

from rag.file.utils.image_utils import compress_image_if_needed
from rag.file.image_caption_tool import ImageCaptionTool
from utils.modelscope_utils import init_mineru_config
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import BlockType, ContentType
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text
from dataclasses import dataclass
import json
from llama_index.core.schema import Document
import tempfile
from loguru import logger


IMAGE_OUTPUT_PREFIX = "mineru_images"
DEFAULT_HEADING_DIFF_THRESHOLD = 2

@dataclass
class TitleInfo:
    text: str
    height: int
    page_idx: int
    idx: int
    bbox: list
    level: int = None


class MineruPdfReader(BaseReader):
    def __init__(
        self, file_store: BaseFileStore, image_caption_tool: ImageCaptionTool = None
    ):
        self.need_init_mineru = True
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool
        logger.info("MineruPdfReader inited.")

    def read(self, file_item: FileItem):
        if self.need_init_mineru:
            init_mineru_config()
            self.need_init_mineru = False
        with tempfile.TemporaryDirectory() as temp_dir:
            local_image_dir = os.path.join(temp_dir, IMAGE_OUTPUT_PREFIX)
            os.makedirs(local_image_dir, exist_ok=True)

            pdf_bytes = file_item.get_data()
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id=0, end_page_id=None)
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(pdf_bytes_list=[new_pdf_bytes],lang_list=['ch','en'])
            )
            model_list = infer_results[0]
            # model_json = copy.deepcopy(model_list)
            image_writer = FileBasedDataWriter(local_image_dir)

            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer,
                _lang, _ocr_enable
            )

            content_list = middle_json["pdf_info"]
            saved_image_map = {}

            if isinstance(self.file_store, OssFileStore):
                image_files = os.listdir(local_image_dir)
                for image_file in image_files:
                    image_path = os.path.join(local_image_dir, image_file)
                    image_save_path = os.path.join(
                        file_item.kb_id, "images", image_file
                    )
                    with open(image_path, "rb") as rf:
                        rf = compress_image_if_needed(rf)
                        if rf:
                            self.file_store.save(rf, image_save_path)
                            saved_image_map[image_file] = image_save_path


            md_content= self.create_markdown(content_list, saved_image_map)

            save_md_file_name = os.path.join(
                file_item.kb_id, "markdown", file_item.file_name + ".md"
            )
            self.file_store.save(BytesIO(md_content.encode("utf-8")), save_md_file_name)

            metadata = file_item.metadata()
            metadata["content_list"] = json.dumps(content_list, ensure_ascii=False)
            return [
                Document(
                    id_=file_item.id,
                    text=md_content,
                    metadata=metadata
                )
            ]

    def create_markdown(
        self,
        pdf_info_dict: list,
        saved_image_map: str = "",
    ):
        output_content = []
        title_list = []
        # 存储每个title的index
        md_title_index = []
        # 记录index
        index_count = 0
        for page_info in pdf_info_dict:
            paras_of_layout = page_info.get("preproc_blocks")
            page_idx = page_info.get("page_idx", -1)
            if not paras_of_layout:
                continue
            (
                page_markdown,
                index_count,
                llm_aided_title,
            ) = self.create_page_markdown(
                paras_of_layout,
                title_list,
                md_title_index,
                page_idx,
                index_count,
                saved_image_map,
            )
            output_content.extend(page_markdown)
        if not llm_aided_title:
            new_title_list = self.post_process_multi_level_headings(
                title_list
            )
        else:
            new_title_list = []
            for index, title_info in enumerate(title_list):
                if not title_info.level or title_info.level < 1:
                    title_info.level = 0
                elif title_info.level > 4:
                    title_info.level = 4
                title_info.level += 1
                title_level = "#" * title_info.level + " "
                new_title = title_level + title_info.text
                new_title_list.append(new_title)
        for idx, content_idx in enumerate(md_title_index):
            output_content[content_idx] = new_title_list[idx]
        markdown_result = "\n\n".join(output_content)
        return markdown_result

    def create_page_markdown(
        self,
        paras_of_layout,
        title_list,
        md_title_index,
        page_idx,
        index_count,
        saved_image_map,
    ):

        page_markdown = []
        llm_aided_title = False
        page_item_index = 0
        for para_block in paras_of_layout:
            para_text = ""
            para_type = para_block["type"]
            if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX]:
                para_text = merge_para_with_text(para_block)
            elif para_type == BlockType.TITLE:
                title_text = merge_para_with_text(para_block)
                para_text = f"# {title_text}"
                md_title_index.append(index_count)
                title_level = para_block.get('level', None)
                bbox = para_block.get('bbox', None)
                if not para_block.get("lines", None) or len(para_block["lines"]) <= 0:
                    title_height = 0
                else:
                    x0, y0, x1, y1 = para_block["lines"][0]["bbox"]
                    title_height = int(y1 - y0)
                title_info = TitleInfo(text=title_text,height=title_height,level=title_level,page_idx=page_idx,idx=page_item_index,bbox=bbox)
                title_list.append(title_info)
                if title_level is not None:
                    llm_aided_title = True
            elif para_type == BlockType.INTERLINE_EQUATION:
                para_text = merge_para_with_text(para_block)
            elif para_type == BlockType.IMAGE:
                for block in para_block["blocks"]:  # 1st.拼image_body
                    if block["type"] == BlockType.IMAGE_BODY:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["type"] == ContentType.IMAGE:
                                    image_path = span.get("image_path", "")
                                    if (
                                        image_path in saved_image_map
                                        and self.image_caption_tool
                                    ):
                                        real_image_path = saved_image_map[image_path]
                                        image_alt_text = (
                                            self.image_caption_tool.extract_url(
                                                self.file_store.get_url(real_image_path)
                                            )
                                        )
                                        para_text += f'\n<img src="{real_image_path}" alt="{image_alt_text}">\n'
                for block in para_block["blocks"]:  # 2nd.拼image_caption
                    if block["type"] == BlockType.IMAGE_CAPTION:
                        para_text += merge_para_with_text(block) + "  \n"
                for block in para_block["blocks"]:  # 3rd.拼image_footnote
                    if block["type"] == BlockType.IMAGE_FOOTNOTE:
                        para_text += merge_para_with_text(block) + "  \n"
            elif para_type == BlockType.TABLE:
                for block in para_block["blocks"]:  # 1st.拼table_caption
                    if block["type"] == BlockType.TABLE_CAPTION:
                        para_text += merge_para_with_text(block) + "  \n"
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
                                    image_path = span.get("image_path", "")
                                    if (
                                        image_path in saved_image_map
                                        and self.image_caption_tool
                                    ):
                                        real_image_path = saved_image_map[image_path]
                                        image_alt_text = (
                                            self.image_caption_tool.extract_url(
                                                self.file_store.get_url(real_image_path)
                                            )
                                        )
                                        para_text += f'\n<img src="{real_image_path}" alt="{image_alt_text}">\n'
                for block in para_block["blocks"]:  # 3rd.拼table_footnote
                    if block["type"] == BlockType.TABLE_FOOTNOTE:
                        para_text += merge_para_with_text(block) + "  \n"

            if para_text.strip() == "":
                continue
            else:
                page_markdown.append(para_text.strip() + "  ")
            page_item_index += 1
            index_count += 1

        return page_markdown, index_count, llm_aided_title

    def average_same_level_title_height(self, title_list):
        groups = defaultdict(list)
        for idx, title_info in enumerate(title_list):
            match = re.match(r"(\d+(\.\d+)*)", title_info.text)
            if match:
                prefix = match.group(1)
                level = prefix.count(".")
                groups[level].append((idx, title_info))

        for titles in groups.values():
            avg_height = int(sum(title_info.height for _, title_info in titles) / len(titles))
            for idx, title_info in titles:
                title_info.height = avg_height
                title_list[idx] = title_info
        return title_list

    def post_process_multi_level_headings(
        self, title_list
    ):
        logger.info(
            "*****************************start process headings*****************************"
        )
        title_list = self.average_same_level_title_height(title_list)
        indexed_title_list = [
            (idx, title_info)
            for idx, title_info in enumerate(title_list)
        ]
        sorted_list = sorted(indexed_title_list, key=lambda item: item[1].height, reverse=True)
        diff_list = [
            (sorted_list[i][1].height - sorted_list[i + 1][1].height, i)
            for i in range(len(sorted_list) - 1)
        ]
        sorted_diff = sorted(diff_list, key=itemgetter(0), reverse=True)
        slice_index = []
        for diff, index in sorted_diff:
            # 标题差的绝对值超过2，则认为是下一级标题
            # markdown 中，# 表示一级标题，## 表示二级标题，以此类推，最多有6级标题，最多能有5次切分
            if diff >= DEFAULT_HEADING_DIFF_THRESHOLD and len(slice_index) <= 5:
                slice_index.append(index)
        slice_index.sort(reverse=True)
        rank_mapping = {}  # idx到rank的映射
        rank = 1
        cur_index = 0
        if len(slice_index) > 0:
            cur_index = slice_index.pop()
        for index, (idx, title_info) in enumerate(sorted_list):
            if index > cur_index:
                rank += 1
                if len(slice_index) > 0:
                    cur_index = slice_index.pop()
                else:
                    cur_index = len(sorted_list) - 1
            rank_mapping[idx] = rank
        new_title_list = []
        for original_idx, title_info in enumerate(title_list):
            assigned_rank = rank_mapping.get(original_idx, 6)
            title_level = "#" * assigned_rank + " "
            title_info.level = assigned_rank

            new_title = title_level + title_info.text
            new_title_list.append(new_title)
            logger.info(f"transform {title_info.text} to {new_title}")

        return new_title_list
