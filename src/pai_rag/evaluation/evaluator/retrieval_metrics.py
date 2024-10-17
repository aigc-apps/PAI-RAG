from typing import List, Optional


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
            # Granular HitRate calculation: Calculate all hits and divide by the number of expected docs
            expected_set = set(expected_ids)
            hits = sum(1 for doc_id in retrieved_ids if doc_id in expected_set)
            score = hits / len(expected_ids) if expected_ids else 0.0
        else:
            # Default HitRate calculation: Check if there is a single hit
            is_hit = any(id in expected_ids for id in retrieved_ids)
            score = 1.0 if is_hit else 0.0

        return score


class MRR:
    """MRR (Mean Reciprocal Rank) metric with two calculation options.

    - The default method calculates the reciprocal rank of the first relevant retrieved document.
    - The more granular method sums the reciprocal ranks of all relevant retrieved documents and divides by the count of relevant documents.

    Attributes:
        metric_name (str): The name of the metric.
        use_granular_mrr (bool): Determines whether to use the granular method for calculation.
    """

    metric_name: str = "mrr"
    use_granular_mrr: bool = False

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
            # Granular MRR calculation: All relevant retrieved docs have their reciprocal ranks summed and averaged
            expected_set = set(expected_ids)
            reciprocal_rank_sum = 0.0
            relevant_docs_count = 0
            for index, doc_id in enumerate(retrieved_ids):
                if doc_id in expected_set:
                    relevant_docs_count += 1
                    reciprocal_rank_sum += 1.0 / (index + 1)
            mrr_score = (
                reciprocal_rank_sum / relevant_docs_count
                if relevant_docs_count > 0
                else 0.0
            )
        else:
            # Default MRR calculation: Reciprocal rank of the first relevant document retrieved
            for i, id in enumerate(retrieved_ids):
                if id in expected_ids:
                    return 1.0 / (i + 1)
            mrr_score = 0.0

        return mrr_score
