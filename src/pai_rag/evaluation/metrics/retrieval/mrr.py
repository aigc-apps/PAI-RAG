from typing import List, Optional
from pai_rag.evaluation.metrics.retrieval.core import (
    default_mrr,
    granular_mrr,
)


class MRR:
    """MRR (Mean Reciprocal Rank) metric with two calculation options.

    - The default method calculates the reciprocal rank of the first relevant retrieved document.
    - The more granular method sums the reciprocal ranks of all relevant retrieved documents and divides by the count of relevant documents.

    Attributes:
        metric_name (str): The name of the metric.
        use_granular_mrr (bool): Determines whether to use the granular method for calculation.
    """

    def __init__(self, metric_name: str = "mrr", use_granular_mrr: bool = False):
        self.metric_name = metric_name
        self.use_granular_mrr = use_granular_mrr

    def compute(
        self,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
    ):
        """Compute MRR based on the provided inputs and selected method.

        Parameters:
            expected_ids (Optional[List[str]]): Expected document IDs.
            retrieved_ids (Optional[List[str]]): Retrieved document IDs.

        Raises:
            ValueError: If the necessary IDs are not provided.

        Returns:
            RetrievalMetricResult: The result with the computed MRR score.
        """
        # Checking for the required arguments
        if (
            retrieved_ids is None
            or expected_ids is None
            or not retrieved_ids
            or not expected_ids
        ):
            raise ValueError("Retrieved ids and expected ids must be provided")

        if self.use_granular_mrr:
            return granular_mrr(expected_ids, retrieved_ids)
        else:
            return default_mrr(expected_ids, retrieved_ids)
