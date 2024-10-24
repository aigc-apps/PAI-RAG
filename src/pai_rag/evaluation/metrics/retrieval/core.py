def default_hit_rate(expected_ids, retrieved_ids):
    """Default HitRate calculation: Check if there is a single hit"""
    is_hit = any(id in expected_ids for id in retrieved_ids)
    score = 1.0 if is_hit else 0.0
    return score


def granular_hit_rate(expected_ids, retrieved_ids):
    """Granular HitRate calculation: Calculate all hits and divide by the number of expected docs"""
    expected_set = set(expected_ids)
    hits = sum(1 for doc_id in retrieved_ids if doc_id in expected_set)
    score = hits / len(expected_ids) if expected_ids else 0.0
    return score


def default_mrr(expected_ids, retrieved_ids):
    """Default MRR calculation: Reciprocal rank of the first relevant document retrieved"""
    for i, id in enumerate(retrieved_ids):
        if id in expected_ids:
            return 1.0 / (i + 1)
    return 0.0


def granular_mrr(expected_ids, retrieved_ids):
    """Granular MRR calculation: All relevant retrieved docs have their reciprocal ranks summed and averaged."""
    expected_set = set(expected_ids)
    reciprocal_rank_sum = 0.0
    relevant_docs_count = 0
    for index, doc_id in enumerate(retrieved_ids):
        if doc_id in expected_set:
            relevant_docs_count += 1
            reciprocal_rank_sum += 1.0 / (index + 1)
    score = (
        reciprocal_rank_sum / relevant_docs_count if relevant_docs_count > 0 else 0.0
    )
    return score
