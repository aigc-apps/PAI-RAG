import os
import cv2
from loguru import logger
from paddleocr import PPStructure
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes


def check_merge_method(in_region):
    """Select the function to merge paragraph.

    Determine the paragraph merging method based on the positional
    relationship between the text bbox and the first line of text in the text bbox.

    Args:
        in_region: Elements with text type in the layout result.

    Returns:
        Merge the functions of paragraph, convert_text_space_head or convert_text_space_tail.
    """
    if len(in_region["res"]) > 0:
        text_bbox = in_region["bbox"]
        text_x1 = text_bbox[0]
        frist_line_box = in_region["res"][0]["text_region"]
        point_1 = frist_line_box[0]
        point_2 = frist_line_box[2]
        frist_line_x1 = point_1[0]
        frist_line_height = abs(point_2[1] - point_1[1])
        x1_distance = frist_line_x1 - text_x1
        return (
            convert_text_space_head
            if x1_distance > frist_line_height
            else convert_text_space_tail
        )


def convert_text_space_head(in_region):
    """The function to merge paragraph.

    The sign of dividing paragraph is that there are two spaces at the beginning.

    Args:
        in_region: Elements with text type in the layout result.

    Returns:
        The text content of the current text box.
    """
    text = ""
    pre_x = None
    frist_line = True
    for i, res in enumerate(in_region["res"]):
        point1 = res["text_region"][0]
        point2 = res["text_region"][2]
        h = point2[1] - point1[1]

        if i == 0:
            text += res["text"]
            pre_x = point1[0]
            continue

        x1 = point1[0]
        if frist_line:
            if abs(pre_x - x1) < h:
                text += "\n\n"
                text += res["text"]
                frist_line = True
            else:
                text += res["text"]
                frist_line = False
        else:
            same_paragh = abs(pre_x - x1) < h
            if same_paragh:
                text += res["text"]
                frist_line = False
            else:
                text += "\n\n"
                text += res["text"]
                frist_line = True
        pre_x = x1
    return text


def convert_text_space_tail(in_region):
    """The function to merge paragraph.

    The symbol for dividing paragraph is a space at the end.

    Args:
        in_region: Elements with text type in the layout result.

    Returns:
        The text content of the current text box.
    """
    text = ""
    frist_line = True
    text_bbox = in_region["bbox"]
    width = text_bbox[2] - text_bbox[0]
    for i, res in enumerate(in_region["res"]):
        point1 = res["text_region"][0]
        point2 = res["text_region"][2]
        row_width = point2[0] - point1[0]
        row_height = point2[1] - point1[1]
        full_row_threshold = width - row_height
        is_full = row_width >= full_row_threshold

        if frist_line:
            text += "\n\n"
            text += res["text"]
        else:
            text += res["text"]

        frist_line = not is_full
    return text


def convert_info_to_text(res, image_name):
    """Save the recognition result as a markdown file.

    Args:
        res: Recognition result
        save_folder: Folder to save the markdown file
        img_name: PDF file or image file name

    Returns:
        None
    """

    text_list = []

    for i, region in enumerate(res):
        merge_func = check_merge_method(region)
        if merge_func:
            text_list.append(merge_func(region))

    text_string = "\n\n".join(text_list)

    logger.info(f"finished processing image {image_name}")
    return text_string


def plain_image_ocr(image_path):
    image_name = os.path.basename(image_path).split(".")[0]
    img = cv2.imread(image_path)
    result = PPStructure(recovery=True)(img)
    _, w, _ = img.shape
    res = sorted_layout_boxes(result, w)
    return convert_info_to_text(res, image_name)
