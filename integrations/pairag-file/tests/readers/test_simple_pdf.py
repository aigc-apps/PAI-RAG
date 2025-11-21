import json
import os
from typing import List, Dict, Any
from loguru import logger
from pairag.file.store.oss_store import OssFileStore
from pairag.file.nodeparsers.file_parser import FileParser
from pairag.file.readers.simple_pdf_reader import SimplePdfReader
from pairag.file.models.file_item import FileItem
import pypdfium2
from PIL import ImageDraw
import pytest

SKIP_CONDITION = not os.environ.get("OSS_ACCESS_KEY_ID") or not os.environ.get("OSS_ACCESS_KEY_SECRET")


# 预定义颜色列表（RGB 格式，范围 [0, 1]）
COLORS = [
    "red",     # 红
    "green",   # 绿
    "yellow",  # 黄
    "blue",    # 蓝
]



def annotate_pdf_with_chunks(pdf_path: str, chunks: list, output_path: str):
    """
    Annotate PDF with bounding boxes from content_list.

    This function renders each PDF page to an image, draws bounding boxes,
    and creates a new PDF from the annotated images.

    Args:
        pdf_path: Path to input PDF
        content_list: List of ContentBlock objects (or dicts)
        output_path: Path to save annotated PDF
    """
    # Group content blocks by page
    pages_content = {}
    for i, chunk in enumerate(chunks):
        logger.info(f"chunk: {chunk.text} {chunk.metadata['token_count']}")
        bbox_list = json.loads(chunk.metadata["page_bbox"])
        for block in bbox_list:
            # Handle both dict and ContentBlock objects
            page = block.get("page_idx")
            bbox = block.get("bbox")
            block["type"] = "text"
            block["chunk_index"] = i

            if page is not None and bbox is not None:
                page_num = page - 1  # Convert to 0-based indexing
                if page_num not in pages_content:
                    pages_content[page_num] = []
                pages_content[page_num].append(block)

    # Open the PDF
    pdf = pypdfium2.PdfDocument(pdf_path)

    # Process each page
    annotated_images = []

    for page_num in range(len(pdf)):
        page = pdf[page_num]

        # Render page to image (scale factor for better quality)
        scale = 2.0  # Higher scale = better quality but larger file
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        # Create a drawing context
        draw = ImageDraw.Draw(pil_image)

        # Draw bounding boxes for content blocks on this page
        if page_num in pages_content:
            for block in pages_content[page_num]:
                # Extract bbox
                if isinstance(block, dict):
                    bbox = block.get("bbox")
                    block_type = block.get("type", "text")
                else:
                    bbox = block.bbox
                    block_type = block.type if hasattr(block, "type") else "text"

                if bbox is None:
                    continue

                x0, y0, x1, y1 = bbox

                # Scale coordinates to match rendered image
                x0_scaled = x0 * scale
                y0_scaled = y0 * scale
                x1_scaled = x1 * scale
                y1_scaled = y1 * scale

                # pdfplumber uses top-left origin, PIL also uses top-left
                # So no coordinate flipping needed

                # Choose color based on content type
                color = COLORS[block["chunk_index"] % len(COLORS)]

                # Draw rectangle
                draw.rectangle([(x0_scaled, y0_scaled), (x1_scaled, y1_scaled)], outline=color, width=2)

        annotated_images.append(pil_image)

    # Create a new PDF from annotated images
    if annotated_images:
        # Save first image as PDF, then append others
        annotated_images[0].save(
            output_path,
            "PDF",
            resolution=100.0,
            save_all=True,
            append_images=annotated_images[1:] if len(annotated_images) > 1 else [],
        )

    pdf.close()
    logger.info(f"Annotated PDF saved to {output_path}")


@pytest.mark.skipif(SKIP_CONDITION, reason="No OSS ak/sk provided")
def test_pdf_reader():
    pdf_file = "tests/testdata/pdf_data/iPhone 16.pdf"
    pdf_file_item = FileItem.from_path(pdf_file, kb_id="test")
    oss_store = OssFileStore(bucket="pai-rag", endpoint="oss-cn-hangzhou.aliyuncs.com")
    file_parser = FileParser(file_store=oss_store)
    docs, chunks = file_parser.parse(pdf_file_item, is_attachment=False)
    logger.info(f"Finished with {len(chunks)} chunks.")

    assert len(chunks) > 0, "No chunks found"

    annotate_pdf_with_chunks(pdf_file, chunks, "tests/testdata/pdf_data/simple_iphone16_annotated.pdf")
