"""LLM reranker."""
from typing import List, Optional

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor import LLMRerank


class MyLLMRerank(LLMRerank):
    """LLM-based reranker.
    Fix the bug : The number of retrieved chunks after LLM Rerank has decreased due to the uncertainty in LLM responses.
    """

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        initial_results: List[NodeWithScore] = []
        for idx in range(0, len(nodes), self.choice_batch_size):
            nodes_batch = [
                node.node for node in nodes[idx : idx + self.choice_batch_size]
            ]

            query_str = query_bundle.query_str
            fmt_batch_str = self._format_node_batch_fn(nodes_batch)
            # call each batch independently
            raw_response = self.llm.predict(
                self.choice_select_prompt,
                context_str=fmt_batch_str,
                query_str=query_str,
            )

            raw_choices, relevances = self._parse_choice_select_answer_fn(
                raw_response, len(nodes_batch)
            )
            choice_idxs = [int(choice) - 1 for choice in raw_choices]
            relevances = relevances
            if len(choice_idxs) < len(nodes):
                missing_numbers = set(range(1, len(nodes))).difference(choice_idxs)
                choice_idxs.extend(missing_numbers)
                relevances.extend([1.0 for _ in missing_numbers])
            choice_nodes = [nodes_batch[idx] for idx in choice_idxs]
            initial_results.extend(
                [
                    NodeWithScore(node=node, score=relevance)
                    for node, relevance in zip(choice_nodes, relevances)
                ]
            )

        return sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[
            : self.top_n
        ]
