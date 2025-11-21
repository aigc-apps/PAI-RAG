from typing import Dict, List, Optional, Tuple
from fastpdf4llm import ContentBlock
from loguru import logger


def classify_fonts(title_sizes: List[float]) -> Dict[float, int]:
    """Classify font sizes into heading levels."""
    logger.info(f"Classifying font sizes: {title_sizes}")
    size_to_level = {}
    unique_sizes = list(dict.fromkeys(title_sizes))

    if not unique_sizes or len(unique_sizes) < 2:
        return size_to_level

    sorted_sizes = sorted(unique_sizes, reverse=True)

    size_ratios = [sorted_sizes[i] / sorted_sizes[i + 1] for i in range(len(sorted_sizes) - 1)]    

    min_diff_ratio = 1.05
    # Identify headers

    last_heading = 1

    size_to_level = {sorted_sizes[0]: 1}
    current_max_size = sorted_sizes[0]
    for i, ratio in enumerate(size_ratios):
        if ratio >= 1.15 and last_heading < 6:
            last_heading += 1
            current_max_size = sorted_sizes[i + 1]
        elif current_max_size >= sorted_sizes[i + 1] * 1.25:
            last_heading += 1
            current_max_size = sorted_sizes[i + 1]
        size_to_level[sorted_sizes[i + 1]] = last_heading

    logger.info(f"Heading sizes to markdown header levels mapping: {size_to_level}")
    return size_to_level


def infer_mineru_api_title_level(content_list: List[ContentBlock]) -> Dict[float, int]:
    title_list = [block for block in content_list if block.text_level is not None and block.type == "text"]
    level_set = set()
    logger.info(f"Inferring title level from content list: {len(title_list)} titles.")
    title_sizes = []
    for title in title_list:
        if title.text and "\n" not in title.text.strip():
            title_sizes.append(title.bbox[3] - title.bbox[1])
            level_set.add(title.text_level)

    if len(level_set) > 1:
        logger.warning(f"Multiple title levels found, skipping title level inference: {level_set}.")
        return
    

    size_to_level = classify_fonts(title_sizes)
    for title in title_list:
        # 1 is default level
        title.text_level = size_to_level.get(title.bbox[3] - title.bbox[1], 1)
        title.text = f"{'#'*title.text_level} {title.text}"
    
    return