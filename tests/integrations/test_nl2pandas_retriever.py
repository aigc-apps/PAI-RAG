import os
import pytest
import pandas as pd

from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core import Settings

from integrations.data_analysis.nl2pandas_retriever import PandasQueryRetriever


dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
llm = DashScope(model_name="qwen-max", temperature=0.1, api_key=dashscope_key)
embed_model = DashScopeEmbedding(embed_batch_size=10, api_key=dashscope_key)
Settings.llm = llm
Settings.embed_model = embed_model


@pytest.mark.skipif(
    os.getenv("DASHSCOPE_API_KEY") is None, reason="no llm api key provided"
)
def test_pandas_query_retriever():
    file_path = "./tests/testdata/csv_data/titanic_train.csv"
    df = pd.read_csv(file_path)
    data_analysis_retriever = PandasQueryRetriever(df)
    query = "What is the correlation between survival and age?"

    retrieved_res = data_analysis_retriever.retrieve(query)

    assert (
        retrieved_res[0].metadata["query_code_instruction"]
        == "df['survived'].corr(df['age'])"
    )

    assert eval(retrieved_res[0].metadata["query_output"]) < 0
