"""
Reranker 测试文件
"""
import pytest
import os
from rag.rerank.dashscope_reranker import DashscopeReranker
from rag.rerank.reranker import RerankResult
import asyncio
from utils.http_session import HttpSessionShared
import aiohttp

@pytest.fixture(scope="session")
def sample_documents():
    """测试用的文档列表 - 关于数据库查询优化的不同相关度文档"""
    return [
        "数据库查询性能优化是提升应用响应速度的关键。可以通过创建合适的索引、优化SQL语句结构、使用查询缓存、分析执行计划等方式来提升查询效率。索引应该建立在经常用于WHERE、JOIN和ORDER BY的列上，但要避免过度索引。",
        "Python是一种高级编程语言，具有简洁的语法和强大的功能。它广泛应用于Web开发、数据分析、人工智能等领域。Python的生态系统非常丰富，有大量的第三方库可以使用。",
        "在MySQL中，可以通过EXPLAIN命令来分析SQL查询的执行计划。执行计划显示了数据库如何执行查询，包括使用的索引、表连接方式等信息。通过分析执行计划，可以找出性能瓶颈并进行优化。",
        "数据库索引是一种数据结构，用于快速定位和访问数据库表中的数据。常见的索引类型包括B树索引、哈希索引等。索引可以显著提高查询速度，但会增加写入操作的开销，因为每次插入、更新或删除数据时都需要维护索引。",
        "Redis是一个开源的内存数据结构存储系统，可以用作数据库、缓存和消息中间件。它支持多种数据结构，如字符串、列表、集合、有序集合等。Redis的读写性能非常高，常用于缓存热点数据。",
        "SQL查询优化技巧包括：避免使用SELECT *，只查询需要的列；使用LIMIT限制返回结果数量；合理使用JOIN，避免笛卡尔积；在WHERE子句中使用索引列；避免在WHERE子句中使用函数，这会导致索引失效。",
        "微服务架构是一种将应用程序构建为一套小型服务的方法，每个服务运行在自己的进程中，并通过轻量级机制（通常是HTTP API）进行通信。这种架构模式有助于提高系统的可扩展性和可维护性。",
    ]


@pytest.fixture(scope="session")
def sample_query():
    """测试用的查询 - 关于数据库查询性能优化的问题"""
    return "如何优化数据库查询性能？有哪些具体的优化方法和技巧？"


@pytest.fixture(scope="session")
def dashscope_reranker():
    """创建 DashscopeReranker 实例"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("需要设置 DASHSCOPE_API_KEY 环境变量")
    
    return DashscopeReranker(
        base_url="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        model="qwen3-rerank",
        timeout=60,
        api_key=api_key
    )


"""DashscopeReranker 重排序效果测试 - 实际 API 调用"""
@pytest.mark.asyncio
async def test_dashscope_rerank(dashscope_reranker, sample_query, sample_documents):
    """测试 rerank 的重排序效果 - 实际调用 DashScope API
    
    验证：
    1. 返回的结果按相关性分数从高到低排序
    2. 最相关的文档（关于数据库查询优化）应该排在前面
    3. 相关性分数是有效的浮点数
    4. 返回结果格式正确
    """

    # 解决session共享问题
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        HttpSessionShared.default = session
        
        results = await dashscope_reranker.rerank(
            query=sample_query,
            documents=sample_documents,
            top_n=3
        )

    # 验证返回结果格式
    assert isinstance(results, list)
    assert len(results) > 0
    
    # 验证每个结果的结构
    for item in results:
        assert isinstance(item, RerankResult)
        assert isinstance(item.index, int)
        assert isinstance(item.score, (int, float))
        assert 0 <= item.score <= 1  # 相关性分数通常在 0-1 之间
        assert isinstance(item.doc, str)
    
    # 验证排序效果：分数应该从高到低
    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True), "结果应该按相关性分数降序排列"

    top_doc_text = sample_documents[results[0].index]
    second_doc_text = sample_documents[results[1].index]
    assert top_doc_text == "数据库查询性能优化是提升应用响应速度的关键。可以通过创建合适的索引、优化SQL语句结构、使用查询缓存、分析执行计划等方式来提升查询效率。索引应该建立在经常用于WHERE、JOIN和ORDER BY的列上，但要避免过度索引。"
    assert second_doc_text == "SQL查询优化技巧包括：避免使用SELECT *，只查询需要的列；使用LIMIT限制返回结果数量；合理使用JOIN，避免笛卡尔积；在WHERE子句中使用索引列；避免在WHERE子句中使用函数，这会导致索引失效。"


# ---------------------------------------------------------------------------
# Unit tests (no real API): vector_store_rerank empty-text filtering
# ---------------------------------------------------------------------------

from unittest.mock import patch, AsyncMock
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult


@pytest.mark.asyncio
async def test_vector_store_rerank_filters_empty_text_nodes():
    """Empty-text nodes must not be sent to DashScope — API rejects them with 400."""
    r = DashscopeReranker(api_key="sk")
    n_empty = TextNode(id_="empty", text="")
    n_keep1 = TextNode(id_="k1", text="real doc 1")
    n_keep2 = TextNode(id_="k2", text="real doc 2")
    vr = VectorStoreQueryResult(
        nodes=[n_empty, n_keep1, n_keep2],
        ids=["empty", "k1", "k2"],
        similarities=[0.1, 0.2, 0.3],
    )
    fake_results = [
        RerankResult(index=1, score=0.9, doc="real doc 2"),
        RerankResult(index=0, score=0.7, doc="real doc 1"),
    ]
    with patch.object(r, "rerank", new=AsyncMock(return_value=fake_results)) as m:
        out = await r.vector_store_rerank(query="q", vector_result=vr, top_n=2)
        kwargs_or_args = m.await_args
        # documents passed positionally as 2nd arg in current impl
        sent_documents = kwargs_or_args.args[1] if len(kwargs_or_args.args) >= 2 else kwargs_or_args.kwargs.get("documents")
        assert "" not in sent_documents
        assert sent_documents == ["real doc 1", "real doc 2"]
        assert out.ids == ["k2", "k1"]


@pytest.mark.asyncio
async def test_vector_store_rerank_all_empty_returns_empty_result():
    r = DashscopeReranker(api_key="sk")
    vr = VectorStoreQueryResult(
        nodes=[TextNode(id_="a", text=""), TextNode(id_="b", text="   ")],
        ids=["a", "b"],
        similarities=[0.1, 0.2],
    )
    with patch.object(r, "rerank", new=AsyncMock()) as m:
        out = await r.vector_store_rerank(query="q", vector_result=vr)
        m.assert_not_called()
    assert out.nodes == []
    assert out.ids == []


@pytest.mark.asyncio
async def test_vector_store_rerank_single_nonempty_after_filter_skips_api():
    r = DashscopeReranker(api_key="sk")
    vr = VectorStoreQueryResult(
        nodes=[TextNode(id_="a", text=""), TextNode(id_="b", text="only one")],
        ids=["a", "b"],
        similarities=[0.1, 0.2],
    )
    with patch.object(r, "rerank", new=AsyncMock()) as m:
        out = await r.vector_store_rerank(query="q", vector_result=vr)
        m.assert_not_called()
    assert out.ids == ["b"]
    # Must preserve original similarity, not invent 1.0
    assert out.similarities == [0.2]


@pytest.mark.asyncio
async def test_vector_store_rerank_single_survivor_preserves_low_score():
    """Regression: an empty-text node with high score must not boost a surviving
    low-score node to 1.0 (which would bypass downstream thresholds and inflate
    multi-KB merge ordering)."""
    r = DashscopeReranker(api_key="sk")
    vr = VectorStoreQueryResult(
        nodes=[
            TextNode(id_="empty_high", text=""),
            TextNode(id_="valid_low", text="real but low"),
        ],
        ids=["empty_high", "valid_low"],
        similarities=[0.99, 0.05],
    )
    with patch.object(r, "rerank", new=AsyncMock()) as m:
        out = await r.vector_store_rerank(query="q", vector_result=vr)
        m.assert_not_called()
    assert out.ids == ["valid_low"]
    assert out.similarities == [0.05]


@pytest.mark.asyncio
async def test_vector_store_rerank_single_empty_node_returns_empty():
    """Single empty-text input must NOT pass through unchanged (pre-filter must run)."""
    r = DashscopeReranker(api_key="sk")
    vr = VectorStoreQueryResult(
        nodes=[TextNode(id_="x", text="")], ids=["x"], similarities=[0.99]
    )
    with patch.object(r, "rerank", new=AsyncMock()) as m:
        out = await r.vector_store_rerank(query="q", vector_result=vr)
        m.assert_not_called()
    assert out.nodes == []
    assert out.similarities == []


@pytest.mark.asyncio
async def test_vector_store_rerank_single_survivor_below_threshold_drops():
    """Sole survivor with score below similarity_threshold must be dropped."""
    r = DashscopeReranker(api_key="sk")
    vr = VectorStoreQueryResult(
        nodes=[
            TextNode(id_="bad", text=""),
            TextNode(id_="low", text="below threshold"),
        ],
        ids=["bad", "low"],
        similarities=[0.99, 0.05],
    )
    with patch.object(r, "rerank", new=AsyncMock()) as m:
        out = await r.vector_store_rerank(
            query="q", vector_result=vr, similarity_threshold=0.8
        )
        m.assert_not_called()
    assert out.nodes == []
    assert out.similarities == []


@pytest.mark.asyncio
async def test_vector_store_rerank_single_input_node_below_threshold_drops():
    """Single non-empty input below threshold must be dropped (no pre-filter bypass)."""
    r = DashscopeReranker(api_key="sk")
    vr = VectorStoreQueryResult(
        nodes=[TextNode(id_="x", text="valid")], ids=["x"], similarities=[0.05]
    )
    with patch.object(r, "rerank", new=AsyncMock()) as m:
        out = await r.vector_store_rerank(
            query="q", vector_result=vr, similarity_threshold=0.8
        )
        m.assert_not_called()
    assert out.nodes == []
