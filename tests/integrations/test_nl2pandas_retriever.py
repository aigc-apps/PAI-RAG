import os
import pytest
import pandas as pd

from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, TextNode

from pai_rag.integrations.data_analysis.nl2pandas_retriever import PandasQueryRetriever
from pai_rag.integrations.data_analysis.data_analysis_synthesizer import (
    DataAnalysisSynthesizer,
)


llm = DashScope(model_name="qwen-max", temperature=0.1)
embed_model = DashScopeEmbedding(embed_batch_size=10)
Settings.llm = llm
Settings.embed_model = embed_model


@pytest.mark.skipif(
    os.getenv("DASHSCOPE_API_KEY") is None, reason="no llm api key provided"
)
def test_pandas_query_retriever():
    file_path = "./tests/testdata/data/csv_data/titanic_train.csv"
    df = pd.read_csv(file_path)
    data_analysis_retriever = PandasQueryRetriever(df)
    query = "What is the correlation between survival and age?"

    retrieved_res = data_analysis_retriever.retrieve(query)

    assert (
        retrieved_res[0].metadata["query_code_instruction"]
        == "df['survived'].corr(df['age'])"
    )

    assert eval(retrieved_res[0].metadata["query_output"]) < 0


@pytest.mark.skipif(
    os.getenv("DASHSCOPE_API_KEY") is None, reason="no llm api key provided"
)
def test_data_analysis_synthesizer():
    query = "What is the correlation between survival and age?"
    retrieved_nodes = [
        NodeWithScore(
            node=TextNode(
                id_="77c9cf14-260f-4d00-9575-aced468a70b6",
                embedding=None,
                metadata={
                    "query_code_instruction": "df['survived'].corr(df['age'])",
                    "query_output": "-0.07722109457217755",
                },
                excluded_embed_metadata_keys=["query_code_instruction", "query_output"],
                excluded_llm_metadata_keys=["query_code_instruction", "query_output"],
                relationships={},
                text="-0.07722109457217755",
                mimetype="text/plain",
                start_char_idx=None,
                end_char_idx=None,
                text_template="{metadata_str}\n\n{content}",
                metadata_template="{key}: {value}",
                metadata_seperator="\n",
            ),
            score=1.0,
        )
    ]
    data_analysis_synthesizer = DataAnalysisSynthesizer()

    res_get_response = data_analysis_synthesizer.get_response(
        query_str=query, retrieved_nodes=retrieved_nodes
    )

    assert len(res_get_response) > 0

    res_synthesize = data_analysis_synthesizer.synthesize(
        query=query, nodes=retrieved_nodes
    )

    assert len(res_synthesize.response) > 0

    assert res_synthesize.source_nodes == retrieved_nodes
