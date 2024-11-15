from typing import List, Optional, Type, Dict
from llama_index.core.bridge.pydantic import Field
import json
from llama_index.core.bridge.pydantic import BaseModel
from pai_rag.evaluation.dataset.rag_qca_dataset import RagQcaSample
from llama_index.core.llama_dataset import CreatedBy
from loguru import logger


class EvaluationSample(RagQcaSample):
    """Response Evaluation RAG example class."""

    hitrate: Optional[float] = Field(
        default_factory=None,
        description="The hitrate value for retrieval evaluation.",
    )
    mrr: Optional[float] = Field(
        default_factory=None,
        description="The mrr value for retrieval evaluation.",
    )

    faithfulness_score: Optional[float] = Field(
        default_factory=None,
        description="The faithfulness score for response evaluation.",
    )

    faithfulness_reason: Optional[str] = Field(
        default_factory=None,
        description="The faithfulness reason for response evaluation.",
    )

    correctness_score: Optional[float] = Field(
        default_factory=None,
        description="The correctness score for response evaluation.",
    )

    correctness_reason: Optional[str] = Field(
        default_factory=None,
        description="The correctness reason for response evaluation.",
    )
    evaluated_by: Optional[CreatedBy] = Field(
        default=None, description="What model generated the evaluation result."
    )

    @property
    def class_name(self) -> str:
        """Data example class name."""
        return "EvaluationSample"


class PaiRagEvalDataset(BaseModel):
    _example_type: Type[EvaluationSample] = EvaluationSample  # type: ignore[misc]
    examples: List[EvaluationSample] = Field(
        default=[], description="Data examples of this dataset."
    )
    results: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Evaluation result of this dataset."
    )
    status: Dict[str, bool] = Field(
        default_factory=dict, description="Status of this dataset."
    )

    @property
    def class_name(self) -> str:
        """Class name."""
        return "PaiRagEvalDataset"

    def cal_mean_metric_score(self) -> float:
        """Calculate the mean metric score."""
        self.results["retrieval"] = {}
        self.results["response"] = {}
        if self.status["retrieval"]:
            self.results["retrieval"] = {
                "mean_hitrate": sum(float(entry.hitrate) for entry in self.examples)
                / len(self.examples),
                "mean_mrr": sum(float(entry.mrr) for entry in self.examples)
                / len(self.examples),
            }
        if self.status["response"]:
            self.results["response"] = {
                "mean_faithfulness_score": sum(
                    float(entry.faithfulness_score) for entry in self.examples
                )
                / len(self.examples),
                "mean_correctness_score": sum(
                    float(entry.correctness_score) for entry in self.examples
                )
                / len(self.examples),
            }

    def save_json(self, path: str) -> None:
        """Save json."""
        self.cal_mean_metric_score()

        with open(path, "w", encoding="utf-8") as f:
            examples = [self._example_type.dict(el) for el in self.examples]
            data = {
                "examples": examples,
                "results": self.results,
                "status": self.status,
            }

            json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved dataset to {path}.")

    @classmethod
    def from_json(cls, path: str) -> "PaiRagEvalDataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)

        examples = [cls._example_type.parse_obj(el) for el in data["examples"]]
        results = data["results"]
        status = data["status"]

        return cls(examples=examples, results=results, status=status)
