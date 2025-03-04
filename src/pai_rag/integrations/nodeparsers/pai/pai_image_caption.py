from pai_rag.integrations.synthesizer.prompt_templates import DEFAULT_EMPTY_RESPONSE_GEN
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.schema import ImageDocument
from typing import Optional, Type
from pai_rag.utils.prompt_template import DEFAULT_IMAGE_CAPTION_PROMPT_ZH
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.core.prompts import PromptTemplate
from llama_index.core.bridge.pydantic_core import CoreSchema, core_schema
from typing import Any
from loguru import logger
import time
from llama_index.core.bridge.pydantic import (
    GetCoreSchemaHandler,
)


class ImageCaption:
    def __init__(
        self,
        multimodal_llm: Optional[MultiModalLLM] = None,
        image_caption_prompt: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__()

        self._multimodal_llm = multimodal_llm
        self._image_caption_prompt = PromptTemplate(
            image_caption_prompt or DEFAULT_IMAGE_CAPTION_PROMPT_ZH
        )

    @classmethod
    def class_name(cls) -> str:
        return "ImageCaption"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.any_schema()

    # TEXT EMBEDDINGS
    def get_image_caption(self, image_url: str) -> RESPONSE_TEXT_TYPE:
        start_time = time.time()
        fmt_prompt = self._image_caption_prompt.format(image_url=image_url)
        llm_response = self._multimodal_llm.complete(
            prompt=fmt_prompt, image_documents=[ImageDocument(image_url=image_url)]
        )
        response = llm_response.text or DEFAULT_EMPTY_RESPONSE_GEN
        end_time = time.time()
        logger.info(
            f"*******Finished ImageCaption for image {image_url} with time {end_time - start_time} seconds"
        )
        return response

    async def aget_image_caption(self, image_url: str) -> RESPONSE_TEXT_TYPE:
        start_time = time.time()
        fmt_prompt = self._image_caption_prompt.format(image_url=image_url)
        llm_response = await self._multimodal_llm.acomplete(
            prompt=fmt_prompt, image_documents=[ImageDocument(image_url=image_url)]
        )
        response = llm_response.text or DEFAULT_EMPTY_RESPONSE_GEN
        end_time = time.time()
        logger.info(
            f"*******Finished ImageCaption for image {image_url} with time {end_time - start_time} seconds"
        )
        return response
