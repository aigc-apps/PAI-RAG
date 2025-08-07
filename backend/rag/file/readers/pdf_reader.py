from collections import defaultdict
from io import BytesIO
from operator import itemgetter
import os
import re
from typing import List
from rag.file.models.file_item import FileItem
from rag.file.readers.base import BaseReader
from rag.file.store.base import BaseFileStore
from rag.file.store.oss_store import OssFileStore

from rag.file.utils.image_utils import compress_image_if_needed
from rag.file.image_caption_tool import ImageCaptionTool
from utils.modelscope_utils import init_mineru_config
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from llama_index.core.schema import Document
from llama_index.llms.openai_like import OpenAILike
import tempfile
from loguru import logger


IMAGE_OUTPUT_PREFIX = "mineru_images"
DEFAULT_HEADING_DIFF_THRESHOLD = 2


class MineruPdfReader(BaseReader):
    def __init__(
        self, file_store: BaseFileStore, image_caption_tool: ImageCaptionTool = None
    ):
        self.need_init_mineru = True
        self.file_store = file_store
        self.image_caption_tool = image_caption_tool
        logger.info("MineruPdfReader inited.")

    def read(self, file_item: FileItem) -> List[Document]:
        if self.need_init_mineru:
            init_mineru_config()
            self.need_init_mineru = False

        with tempfile.TemporaryDirectory() as temp_dir:
            local_image_dir = os.path.join(temp_dir, IMAGE_OUTPUT_PREFIX)
            os.makedirs(local_image_dir, exist_ok=True)

            image_writer = FileBasedDataWriter(local_image_dir)
            pdf_bytes = file_item.get_data()
            ds = PymuDocDataset(pdf_bytes)

            # 选择解析方式
            if ds.classify() == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

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

            content_list = pipe_result._pipe_res["pdf_info"]

            md_content = self.create_markdown(content_list, saved_image_map)

            save_md_file_name = os.path.join(
                file_item.kb_id, "markdown", file_item.file_name + ".md"
            )
            self.file_store.save(BytesIO(md_content.encode("utf-8")), save_md_file_name)

            metadata = file_item.metadata()
            return [
                Document(
                    id_=file_item.id,
                    text=md_content,
                    metadata=metadata,
                )
            ]

    def create_markdown(
        self,
        pdf_info_dict: list,
        saved_image_map: str = "",
    ):
        output_content = []
        text_height_min = float("inf")
        text_height_max = 0
        title_list = []
        # 存储每个title的index
        md_title_index = []
        # 记录index
        index_count = 0
        for page_info in pdf_info_dict:
            paras_of_layout = page_info.get("para_blocks")
            if not paras_of_layout:
                continue
            (
                page_markdown,
                text_height_min,
                text_height_max,
                index_count,
            ) = self.create_page_markdown(
                paras_of_layout,
                title_list,
                md_title_index,
                text_height_min,
                text_height_max,
                index_count,
                saved_image_map,
            )
            output_content.extend(page_markdown)
        new_title_list = self.post_process_multi_level_headings(
            title_list, text_height_min, text_height_max
        )
        for idx, content_idx in enumerate(md_title_index):
            output_content[content_idx] = new_title_list[idx]
        markdown_result = "\n\n".join(output_content)
        return markdown_result

    def create_page_markdown(
        self,
        paras_of_layout,
        title_list,
        md_title_index,
        text_height_min,
        text_height_max,
        index_count,
        saved_image_map,
    ):
        from magic_pdf.dict2md.ocr_mkcontent import merge_para_with_text

        page_markdown = []
        for para_block in paras_of_layout:
            text_height_min, text_height_max = self.collect_title_info(
                para_block, title_list, text_height_min, text_height_max
            )
            para_text = ""
            para_type = para_block["type"]
            if para_type in [BlockType.Text, BlockType.List, BlockType.Index]:
                para_text = merge_para_with_text(para_block)
            elif para_type == BlockType.Title:
                para_text = f"# {merge_para_with_text(para_block)}"
                md_title_index.append(index_count)
            elif para_type == BlockType.InterlineEquation:
                para_text = merge_para_with_text(para_block)
            elif para_type == BlockType.Image:
                for block in para_block["blocks"]:  # 1st.拼image_body
                    if block["type"] == BlockType.ImageBody:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["type"] == ContentType.Image:
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
                    if block["type"] == BlockType.ImageCaption:
                        para_text += merge_para_with_text(block) + "  \n"
                for block in para_block["blocks"]:  # 3rd.拼image_footnote
                    if block["type"] == BlockType.ImageFootnote:
                        para_text += merge_para_with_text(block) + "  \n"
            elif para_type == BlockType.Table:
                for block in para_block["blocks"]:  # 1st.拼table_caption
                    if block["type"] == BlockType.TableCaption:
                        para_text += merge_para_with_text(block) + "  \n"
                for block in para_block["blocks"]:  # 2nd.拼table_body
                    if block["type"] == BlockType.TableBody:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span["type"] == ContentType.Table:
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
                    if block["type"] == BlockType.TableFootnote:
                        para_text += merge_para_with_text(block) + "  \n"

            if para_text.strip() == "":
                continue
            else:
                page_markdown.append(para_text.strip() + "  ")
            index_count += 1

        return page_markdown, text_height_min, text_height_max, index_count

    def collect_title_info(
        self, para_block, title_list, text_height_min, text_height_max
    ):
        from magic_pdf.dict2md.ocr_mkcontent import merge_para_with_text

        if not para_block.get("lines", None) or len(para_block["lines"]) <= 0:
            return text_height_min, text_height_max
        x0, y0, x1, y1 = para_block["lines"][0]["bbox"]
        content_height = y1 - y0
        if para_block["type"] == BlockType.Title:
            title_height = int(content_height)
            title_text = merge_para_with_text(para_block)
            title_list.append((title_text, title_height))
        elif para_block["type"] == BlockType.Text:
            if content_height < text_height_min:
                text_height_min = content_height
            if content_height > text_height_max:
                text_height_max = content_height
        return text_height_min, text_height_max

    def average_same_level_title_height(self, title_list):
        groups = defaultdict(list)
        for idx, (title_text, title_height) in enumerate(title_list):
            match = re.match(r"(\d+(\.\d+)*)", title_text)
            if match:
                prefix = match.group(1)
                level = prefix.count(".")
                groups[level].append((idx, title_text, title_height))

        for titles in groups.values():
            avg_height = int(sum(height for _, _, height in titles) / len(titles))
            for idx, title_text, _ in titles:
                title_list[idx] = (title_text, avg_height)
        return title_list

    def post_process_multi_level_headings(
        self, title_list, text_height_min, text_height_max
    ):
        logger.info(
            "*****************************start process headings*****************************"
        )
        title_list = self.average_same_level_title_height(title_list)
        indexed_title_list = [
            (idx, title_text, title_height)
            for idx, (title_text, title_height) in enumerate(title_list)
        ]
        sorted_list = sorted(indexed_title_list, key=itemgetter(2), reverse=True)
        diff_list = [
            (sorted_list[i][2] - sorted_list[i + 1][2], i)
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
        for index, (idx, title_text, title_height) in enumerate(sorted_list):
            if index > cur_index:
                rank += 1
                if len(slice_index) > 0:
                    cur_index = slice_index.pop()
                else:
                    cur_index = len(sorted_list) - 1
            rank_mapping[idx] = rank
        new_title_list = []
        for original_idx, (title_text, title_height) in enumerate(title_list):
            assigned_rank = rank_mapping.get(original_idx, 6)
            title_level = "#" * assigned_rank + " "

            # 高度范围在纯文本之间不作为标题
            if text_height_min <= text_height_max and int(
                text_height_min
            ) <= title_height <= int(text_height_max):
                title_level = ""

            new_title = title_level + title_text
            new_title_list.append(new_title)
            logger.info(f"transform {title_text} to {new_title}")

        return new_title_list


if __name__ == "__main__":
    pdf_file = "/Users/feiyue/Documents/test_files/1.pdf"
    pdf_file_item = FileItem.from_path(pdf_file, knowledgebase_id="test")
    multimodal_llm = OpenAILike(
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        model="qwen-vl-max",
        is_chat_model=True,
    )
    image_caption_tool = ImageCaptionTool(multimodal_llm=multimodal_llm)
    oss_store = OssFileStore(bucket="pai-rag", endpoint="oss-cn-hangzhou.aliyuncs.com")
    pdf_reader = MineruPdfReader(
        file_store=oss_store, image_caption_tool=image_caption_tool
    )
    doc = pdf_reader.read(pdf_file_item)
    print("finished.")
