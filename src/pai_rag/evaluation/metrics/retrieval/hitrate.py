from typing import List, Optional
from pai_rag.evaluation.metrics.retrieval.core import (
    granular_hit_rate,
    default_hit_rate,
)


class HitRate:
    """Hit rate metric: Compute hit rate with two calculation options.

    - The default method checks for a single match between any of the retrieved docs and expected docs.
    - The more granular method checks for all potential matches between retrieved docs and expected docs.

    Attributes:
        metric_name (str): The name of the metric.
        use_granular_hit_rate (bool): Determines whether to use the granular method for calculation.
    """

    metric_name: str = "hitrate"
    use_granular_hit_rate: bool = False

    def compute(
        self,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
    ):
        """Compute metric based on the provided inputs.

        Parameters:
            expected_ids (Optional[List[str]]): Expected document IDs.
            retrieved_ids (Optional[List[str]]): Retrieved document IDs.

        Raises:
            ValueError: If the necessary IDs are not provided.

        Returns:
            RetrievalMetricResult: The result with the computed hit rate score.
        """
        # Checking for the required arguments
        if (
            retrieved_ids is None
            or expected_ids is None
            or not retrieved_ids
            or not expected_ids
        ):
            raise ValueError("Retrieved ids and expected ids must be provided")

        if self.use_granular_hit_rate:
            return granular_hit_rate(expected_ids, retrieved_ids)
        else:
            return default_hit_rate(expected_ids, retrieved_ids)
