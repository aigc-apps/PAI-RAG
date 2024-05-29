import logging
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry
from llama_index.core.prompts.prompt_type import PromptType

# from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from pai_rag.evaluations.dataset_generation.ragdataset_generator import (
    ModifiedRagDatasetGenerator,
)
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.prompts.base import PromptTemplate
from typing import Optional
from pai_rag.utils.prompt_template import (
    DEFAULT_QUESTION_GENERATION_PROMPT,
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_QUESTION_GENERATION_QUERY,
)
import json


class GenerateDatasetPipeline(ModifiedRagDatasetGenerator):
    def __init__(
        self,
        text_question_template_str: Optional[str] = DEFAULT_QUESTION_GENERATION_PROMPT,
        text_qa_template_str: Optional[str] = DEFAULT_TEXT_QA_PROMPT_TMPL,
        question_gen_query: Optional[str] = DEFAULT_QUESTION_GENERATION_QUERY,
        num_questions_per_chunk: int = 1,
        show_progress: Optional[bool] = True,
    ) -> None:
        self.name = "GenerateDatasetPipeline"
        self.nodes = list(
            module_registry.get_module("IndexModule").docstore.docs.values()
        )
        self.num_questions_per_chunk = num_questions_per_chunk
        self.llm = module_registry.get_module("LlmModule")
        self.text_question_template = PromptTemplate(text_question_template_str)
        self.text_qa_template = PromptTemplate(
            text_qa_template_str, prompt_type=PromptType.QUESTION_ANSWER
        )
        self.question_gen_query = question_gen_query.format(
            num_questions_per_chunk=self.num_questions_per_chunk
        )
        self.show_progress = show_progress

        logging.info("dataset generation initialized successfully.")

    def generate_dataset(self) -> LabelledRagDataset:
        dataset_generator = ModifiedRagDatasetGenerator(
            nodes=self.nodes,
            llm=self.llm,
            num_questions_per_chunk=self.num_questions_per_chunk,
            text_question_template=self.text_question_template,
            text_qa_template=self.text_qa_template,
            question_gen_query=self.question_gen_query,
            show_progress=self.show_progress,
        )

        qas = dataset_generator.generate_dataset_from_nodes()
        logging.info("dataset generation completed.")
        return qas

    async def agenerate_dataset(self) -> LabelledRagDataset:
        dataset_generator = ModifiedRagDatasetGenerator(
            nodes=self.nodes,
            llm=self.llm,
            num_questions_per_chunk=self.num_questions_per_chunk,
            text_question_template=self.text_question_template,
            text_qa_template=self.text_qa_template,
            question_gen_query=self.question_gen_query,
            show_progress=self.show_progress,
        )
        qas = await dataset_generator.agenerate_dataset_from_nodes()
        logging.info("dataset generation completed.")
        return qas

    def save_json(self, qas: LabelledRagDataset, path: str) -> None:
        """Save json."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(qas.dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> LabelledRagDataset:
        """Load json."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data


if __name__ == "__main__":
    config = RagConfiguration.from_file("dummy_path").get_value()
    module_registry.init_modules(config)
    pipeline = GenerateDatasetPipeline()
    qas = pipeline.generate_dataset()
    print("nodes_id:", [node.node_id for node in pipeline.nodes[0:2]])
    save_file_path = "data/qa_dataset_tmp.json"
    pipeline.save_json(qas, save_file_path)
