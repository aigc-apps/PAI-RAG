import logging
import os
from pathlib import Path
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

_BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_EVAL_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")
DEFAULT_EVAL_DATA_FOLDER = "tests/testdata/paul_graham"


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
        self.config = RagConfiguration.from_file(DEFAULT_EVAL_CONFIG_FILE).get_value()

        # load nodes
        module_registry.init_modules(self.config)
        datareader_factory = module_registry.get_module_with_config(
            "DataReaderFactoryModule", self.config
        )
        self.node_parser = module_registry.get_module_with_config(
            "NodeParserModule", self.config
        )
        reader = datareader_factory.get_reader(DEFAULT_EVAL_DATA_FOLDER)
        docs = reader.load_data()
        self.nodes = self.node_parser.get_nodes_from_documents(docs)

        self.num_questions_per_chunk = num_questions_per_chunk
        self.llm = module_registry.get_module_with_config("LlmModule", self.config)
        self.text_question_template = PromptTemplate(text_question_template_str)
        self.text_qa_template = PromptTemplate(
            text_qa_template_str, prompt_type=PromptType.QUESTION_ANSWER
        )
        self.question_gen_query = question_gen_query.format(
            num_questions_per_chunk=self.num_questions_per_chunk
        )
        self.show_progress = show_progress
        self.is_test_run = os.getenv("IS_PAI_RAG_CI_TEST") == "true"
        if self.is_test_run:
            self.nodes = self.nodes[:1]  # Only

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
