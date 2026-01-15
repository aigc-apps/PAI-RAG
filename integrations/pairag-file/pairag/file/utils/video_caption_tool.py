from typing import Optional
from loguru import logger
import base64

from pairag.file.utils.multimodal_llm import OpenAIMultimodalLLM


system_prompt_str = """
你是一个视频处理专家，善于提取视频里的有用信息，给视频生成简洁完整的描述。
"""

# Video MIME type mapping
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
    ".3gp": "video/3gpp",
}


def get_video_mime_type(file_extension: str) -> str:
    """
    Get the MIME type for a video file extension.
    
    Args:
        file_extension: File extension (e.g., '.mp4', '.avi')
        
    Returns:
        MIME type string (e.g., 'video/mp4')
    """
    ext = file_extension.lower() if file_extension.startswith('.') else f'.{file_extension.lower()}'
    return VIDEO_MIME_TYPES.get(ext, "video/mp4")  # Default to mp4


def encode_video_to_data_url(video_data: bytes, file_extension: str = ".mp4") -> str:
    """
    Encode video data to a base64 data URL with proper MIME type prefix.
    
    Args:
        video_data: Raw video bytes
        file_extension: File extension to determine MIME type
        
    Returns:
        Data URL string (e.g., 'data:video/mp4;base64,...')
    """
    mime_type = get_video_mime_type(file_extension)
    video_base64 = base64.b64encode(video_data).decode('utf-8')
    return f"data:{mime_type};base64,{video_base64}"


class VideoCaptionTool:
    def __init__(self, multimodal_llm: OpenAIMultimodalLLM):
        assert (
            multimodal_llm is not None
        ), "Must provide a multimodal_llm for the video captioning tool."

        self.multimodal_llm = multimodal_llm

    def extract_video(self, video_data: bytes, file_extension: str = ".mp4") -> Optional[str]:
        """
        Run the video captioning model on the given video data.
        
        Args:
            video_data: Raw video bytes
            file_extension: File extension (e.g., '.mp4', '.avi') for MIME type detection
            
        Returns:
            Video caption/description string, or None if no content detected
        """
        logger.info(f"[视频解析] 正在解析视频, 文件类型: {file_extension}")
        
        # Encode video with proper MIME type prefix
        video_data_url = encode_video_to_data_url(video_data, file_extension)

        caption_result = self.multimodal_llm.chat_with_video(
            system_prompt=system_prompt_str,
            video_urls=[video_data_url],
        )

        logger.info(f"[视频解析] 解析视频结果: {caption_result}")
        if not caption_result or "NO_VIDEO_CONTENT" in caption_result:
            return None

        return caption_result
