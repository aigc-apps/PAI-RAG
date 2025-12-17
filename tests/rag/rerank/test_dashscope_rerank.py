"""
Reranker 测试文件
"""
import pytest
import os
from rag.rerank.dashscope_reranker import DashscopeReranker
from rag.rerank.reranker import RerankResult
from conftest import event_loop

    

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
def test_dashscope_rerank(event_loop, dashscope_reranker, sample_query, sample_documents):
    """测试 rerank 的重排序效果 - 实际调用 DashScope API
    
    验证：
    1. 返回的结果按相关性分数从高到低排序
    2. 最相关的文档（关于数据库查询优化）应该排在前面
    3. 相关性分数是有效的浮点数
    4. 返回结果格式正确
    """
    pytest.skip("跳过，目前有asyncio错误")

    results = event_loop.run_until_complete(dashscope_reranker.rerank(
        query=sample_query,
        documents=sample_documents,
        top_n=3
    ))

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

