from pai.llm_eval.evals.default_templates import DefaultPromptTemplateCN
from pai.llm_eval.pipeline.pipeline_utils import run_rag_offline_eval_pipeline
from pai_rag.evaluation.evaluator.base_evaluator import BaseEvaluator
from pai_rag.evaluation.dataset.rag_qca_dataset_refactor import QcapDataset
from pai_rag.evaluation.dataset.state_manager import (
    DatasetState,
    StateManager,
)
from pai_rag.evaluation.dataset.rag_eval_dataset_refactor import (
    QcapEvalDataset,
)
from loguru import logger


class PaiEvaluator(BaseEvaluator):
    def __init__(
        self,
        llm_config,
        persist_path: str = None,
        state_manager: StateManager = None,
    ):
        super().__init__(persist_path=persist_path, state_manager=state_manager)

        self._llm_config = llm_config
        self.retrieval_evaluators = [
            DefaultPromptTemplateCN.RETRIEVER_RELEVANCE_PROMPT_TEMPLATE
        ]
        self.response_evaluators = [
            DefaultPromptTemplateCN.FAITHFULNESS_PROMPT_TEMPLATE,
            DefaultPromptTemplateCN.RAG_CORRECTNESS_PROMPT_TEMPLATE,
        ]

    async def aevaluation_for_retrieval(self):
        if self.state_manager.is_completed(DatasetState.RETRIEVAL):
            logger.info("Evaluation dataset for retrieval stage already exists.")
        else:
            logger.info(
                "Starting to generate evaluation dataset for retrieval stage..."
            )
            qcap_dataset = QcapDataset.from_json(self.qcap_dataset_path)
            ########################################################
            # TODO: update to new sdk
            retrieval_eval_samples = run_rag_offline_eval_pipeline(
                self.retrieval_evaluators,
                qcap_dataset,
                eval_name="test_rag_offline_eval_retrieval",
                batch_size=1,
                need_data_management=False,
                **self._llm_config,
            )
            ########################################################
            retrieval_eval_dataset = QcapEvalDataset(samples=retrieval_eval_samples)
            retrieval_eval_dataset.save_json(self.retrieval_evaluation_dataset_path)
            self.state_manager.mark_completed(DatasetState.RETRIEVAL)
            logger.info("Evaluation dataset for retrieval stage generated.")

    async def aevaluation_for_response(self):
        if self.state_manager.is_completed(DatasetState.RESPONSE):
            logger.info("Evaluation dataset for response stage already exists.")
        else:
            logger.info("Starting to generate evaluation dataset for response stage...")
            qcap_dataset = QcapDataset.from_json(self.qcap_dataset_path)
            ########################################################
            # TODO: update to new sdk
            response_eval_samples = run_rag_offline_eval_pipeline(
                self.response_evaluators,
                qcap_dataset,
                eval_name="test_rag_offline_eval_response",
                batch_size=1,
                need_data_management=False,
                **self._llm_config,
            )
            ########################################################
            response_eval_dataset = QcapEvalDataset(samples=response_eval_samples)
            response_eval_dataset.save_json(self.response_evaluation_dataset_path)
            self.state_manager.mark_completed(DatasetState.RESPONSE)
            logger.info("Evaluation dataset for response stage generated.")

    async def aevaluation_for_all(self):
        """Run evaluation with qca dataset."""
        if self.state_manager.is_completed(DatasetState.E2E):
            logger.info("Evaluation dataset for end2end stage already exists.")
        else:
            logger.info("Starting to generate evaluation dataset for end2end stage...")
            qcap_dataset = QcapDataset.from_json(self.qcap_dataset_path)
            ########################################################
            # TODO: update to new sdk
            e2e_eval_samples = run_rag_offline_eval_pipeline(
                self.response_evaluators,
                qcap_dataset,
                eval_name="test_rag_offline_eval_all",
                batch_size=1,
                need_data_management=False,
                **self._llm_config,
            )
            ########################################################
            e2e_eval_dataset = QcapEvalDataset(samples=e2e_eval_samples)
            e2e_eval_dataset.save_json(self.evaluation_dataset_path)
            self.state_manager.mark_completed(DatasetState.E2E)
            logger.info("Evaluation dataset for end2end stage generated.")
