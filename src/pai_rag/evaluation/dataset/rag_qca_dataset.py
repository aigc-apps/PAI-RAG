from typing import List, Optional, Type
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llama_dataset.base import BaseLlamaDataExample
from llama_index.core.llama_dataset import CreatedBy
import json
from llama_index.core.bridge.pydantic import BaseModel


class RagQcaSample(BaseLlamaDataExample):
    """Predicted RAG example class. Analogous to traditional ML datasets, this dataset contains
    the "features" (i.e., query + context) to make a prediction and the "label" (i.e., response)
    to evaluate the prediction.
    """

    query: str = Field(
        default_factory=str, description="The user query for the example."
    )
    query_by: Optional[CreatedBy] = Field(
        default=None, description="What generated the query."
    )
    reference_contexts: Optional[List[str]] = Field(
        default_factory=None,
        description="The contexts used to generate the reference answer.",
    )
    reference_node_id: Optional[List[str]] = Field(
        default_factory=None, description="The node id corresponding to the contexts"
    )
    reference_answer: str = Field(
        default_factory=str,
        description="The reference (ground-truth) answer to the example.",
    )
    reference_answer_by: Optional[CreatedBy] = Field(
        default=None, description="What model generated the reference answer."
    )

    predicted_contexts: Optional[List[str]] = Field(
        default_factory=None,
        description="The contexts used to generate the predicted answer.",
    )
    predicted_node_id: Optional[List[str]] = Field(
        default_factory=None,
        description="The node id corresponding to the predicted contexts",
    )
    predicted_answer: str = Field(
        default_factory=str,
        description="The predicted answer to the example.",
    )
    predicted_answer_by: Optional[CreatedBy] = Field(
        default=None, description="What model generated the predicted answer."
    )

    @property
    def class_name(self) -> str:
        """Data example class name."""
        return "RagQcaSample"


class PaiRagQcaDataset(BaseModel):
    _example_type: Type[RagQcaSample] = RagQcaSample  # type: ignore[misc]
    examples: List[RagQcaSample] = Field(
        default=[], description="Data examples of this dataset."
    )
    labelled: bool = Field(
        default=False, description="Whether the dataset is labelled or not."
    )
    predicted: bool = Field(
        default=False, description="Whether the dataset is predicted or not."
    )

    @property
    def class_name(self) -> str:
        """Class name."""
        return "PaiRagQcaDataset"

    def save_json(self, path: str) -> None:
        """Save json."""
        with open(path, "w", encoding="utf-8") as f:
            examples = [self._example_type.dict(el) for el in self.examples]
            data = {
                "examples": examples,
                "labelled": self.labelled,
                "predicted": self.predicted,
            }

            json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Saved PaiRagQcaDataset to {path}.")

    @classmethod
    def from_json(cls, path: str) -> "PaiRagQcaDataset":
        """Load json."""
        with open(path) as f:
            data = json.load(f)

        if len(data["examples"]) > 0:
            examples = [cls._example_type.parse_obj(el) for el in data["examples"]]
            labelled = data["labelled"]
            predicted = data["predicted"]

            return cls(examples=examples, labelled=labelled, predicted=predicted)
        else:
            return None
