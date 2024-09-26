"""synthesizer factory, used to generate synthesizer instance based on customer config"""

import logging
from typing import Dict, List, Any

from llama_index.core.prompts import (
    PromptTemplate,
)
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_TEXT_QA_PROMPT_SEL,
)
from pai_rag.integrations.synthesizer.pai_synthesizer import (
    DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL,
    PaiSynthesizer,
)
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.utils.prompt_template import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
)


logger = logging.getLogger(__name__)


class SynthesizerModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["LlmModule", "MultiModalLlmModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        llm = new_params["LlmModule"]
        multimodal_llm = new_params["MultiModalLlmModule"]

        text_qa_template_str = config.get(
            "text_qa_template", DEFAULT_TEXT_QA_PROMPT_TMPL
        )
        if text_qa_template_str:
            text_qa_template = PromptTemplate(text_qa_template_str)
        else:
            text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL

        multimodal_qa_template_str = config.get(
            "multimodal_qa_template", DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL
        )
        multimodal_qa_template = None
        if multimodal_qa_template_str:
            multimodal_qa_template = PromptTemplate(multimodal_qa_template_str)

        return PaiSynthesizer(
            llm=llm,
            text_qa_template=text_qa_template,
            multimodal_llm=multimodal_llm,
            multimodal_qa_template=multimodal_qa_template,  # Customize qa template
        )
