from typing import Dict
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from loguru import logger

from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_iqs20241111 import models
from alibabacloud_iqs20241111.client import Client
from Tea.exceptions import TeaException

from pairag.integrations.search.bing_search import DEFAULT_SEARCH_COUNT
from pairag.integrations.search.search_config import (
    DEFAULT_ALIYUN_SEARCH_ENDPOINT,
)
import time

DEFAULT_LANG = "zh-CN"
DEFAULT_TIMERANGE = "OneYear"  # OneMonth, OneWeek, OneDay, OneYear, NoLimit
DEFAULT_ENGINE_TYPE = "LiteAdvanced"  # Generic, GenericAdvanced, LiteAdvanced, Deep
DEFAULT_MAX_RESULTS = 20
DEFAULT_MIN_RESULTS = 5

# 支持的搜索引擎类型
SUPPORTED_ENGINE_TYPES = ["Generic", "GenericAdvanced", "LiteAdvanced", "Deep"]


def should_exclude(tags: Dict[str, str]) -> bool:
    """
    基于阿里云搜索API Tags判断是否应该排除该结果

    参考API文档，采用排除逻辑过滤低质量内容：
    1. 低质量UGC内容：ugcType in ("StructuredQA", "NoteShare", "ForumPost")
    2. 论坛UGC：isUgc == "true" && genre == "ForumUgc"
    3. 列表页：isListPage == "true"
    4. 可选：商业类内容（根据业务需求决定是否启用）
    """
    # 根据API文档的枚举值进行过滤
    exclude_ugc_types = ("StructuredQA", "NoteShare", "ForumPost")

    ugc_type = tags.get("ugcType", "")
    genre = tags.get("genre", "")
    is_ugc = tags.get("isUgc", "")
    is_list_page = tags.get("isListPage", "")

    # 条件1: 过滤低质量UGC类型（问答、笔记、论坛帖子）
    if ugc_type in exclude_ugc_types:
        return True

    # 条件2: 过滤论坛UGC内容
    if is_ugc == "true" and genre == "ForumUgc":
        return True

    # 条件3: 过滤列表页面（通常质量较低）
    if is_list_page == "true":
        return True

    # 条件4: 可选 - 过滤商业类内容（根据具体场景决定）
    # if genre == "Commerce":
    #     return True

    return False


def _get_exclude_reason(tags: Dict[str, str]) -> str:
    """获取排除原因说明"""
    ugc_type = tags.get("ugcType", "")
    genre = tags.get("genre", "")
    is_ugc = tags.get("isUgc", "")
    is_list_page = tags.get("isListPage", "")

    reasons = []

    if ugc_type in ("StructuredQA", "NoteShare", "ForumPost"):
        reasons.append(f"低质量UGC类型: {ugc_type}")

    if is_ugc == "true" and genre == "ForumUgc":
        reasons.append("论坛UGC内容")

    if is_list_page == "true":
        reasons.append("列表页面")

    if genre == "Commerce":
        reasons.append("商业类内容")

    return "; ".join(reasons)


def optimize_search_results(
    results: Dict,
    min_results: int = DEFAULT_MIN_RESULTS,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> Dict:
    """
    基于Tags标签优化搜索结果

    采用排除逻辑过滤低质量内容，当过滤后结果不足时自动降级为全量召回

    Args:
        results: 原始搜索结果字典
        min_results: 最少保留结果数，低于此数量时降级为全量召回
        max_results: 最多保留结果数，超过此数量时强制截断

    Returns:
        优化后的搜索结果字典
    """
    page_items = results.get("pageItems", [])
    if not page_items:
        return results

    # 过滤逻辑
    filtered_items = []
    excluded_items = []

    for item in page_items:
        if len(filtered_items) >= max_results:
            break

        tags = item.get("tags", {})

        # 如果没有tags信息，默认保留（避免误过滤）
        if not tags:
            filtered_items.append(item)
            continue

        if should_exclude(tags):
            excluded_items.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "tags": tags,
                    "exclude_reason": _get_exclude_reason(tags),
                }
            )
        else:
            filtered_items.append(item)

    # 降级策略：如果过滤后结果太少，则使用原始结果
    degraded = len(filtered_items) < min_results
    if degraded:
        logger.info(
            f"搜索结果优化降级：过滤后结果数({len(filtered_items)}) < 最少保留数({min_results})，使用原始结果"
        )
        filtered_items = page_items[:max_results]  # 限制最大数量
        excluded_items = []

    # 构建优化后的结果
    optimized_results = results.copy()
    optimized_results["pageItems"] = filtered_items

    # 添加优化统计信息
    optimization_stats = {
        "original_count": len(page_items),
        "filtered_count": len(filtered_items),
        "excluded_count": len(excluded_items),
        "degraded": degraded,
        "excluded_items": excluded_items[:3] if excluded_items else [],  # 只记录前3个排除项
    }

    # 记录优化日志
    logger.info(
        f"搜索结果优化完成: 原始{optimization_stats['original_count']}条 -> "
        f"过滤{optimization_stats['filtered_count']}条, "
        f"排除{optimization_stats['excluded_count']}条, "
        f"降级: {'是' if degraded else '否'}"
    )

    optimized_results["optimization"] = optimization_stats
    return optimized_results


class AliyunSearchTool(BaseRetriever):
    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        endpoint: str = DEFAULT_ALIYUN_SEARCH_ENDPOINT,
        search_count: int = DEFAULT_SEARCH_COUNT,
        search_lang: str = DEFAULT_LANG,
        time_range: str = DEFAULT_TIMERANGE,
        engine_type: str = DEFAULT_ENGINE_TYPE,
        enable_optimization: bool = True,
        min_results: int = DEFAULT_MIN_RESULTS,
        max_results: int = DEFAULT_MAX_RESULTS,
    ):
        """
        初始化阿里云搜索工具

        Args:
            access_key_id: 阿里云 Access Key ID
            access_key_secret: 阿里云 Access Key Secret
            endpoint: 搜索服务端点
            search_count: 搜索结果数量
            search_lang: 搜索语言
            time_range: 时间范围 (OneMonth, OneWeek, OneDay, OneYear, NoLimit)
            engine_type: 搜索引擎类型 (Generic, GenericAdvanced, LiteAdvanced, Deep)
            enable_optimization: 是否启用基于Tags的结果优化
            min_results: 优化时最少保留结果数
            max_results: 优化时最多保留结果数
        """
        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
        )

        self.search_count = search_count
        self.search_lang = search_lang
        self.engine_type = (
            engine_type
            if engine_type in SUPPORTED_ENGINE_TYPES
            else DEFAULT_ENGINE_TYPE
        )
        self.enable_optimization = enable_optimization
        self.min_results = min_results
        self.max_results = max_results

        config.endpoint = endpoint
        self.Client = Client(config)
        self.time_range = time_range

        logger.info(
            f"初始化阿里云搜索工具: endpoint={endpoint}, engine_type={self.engine_type}, "
            f"optimization={'启用' if self.enable_optimization else '禁用'}, "
            f"time_range={time_range}, search_count={search_count}"
        )

    async def _search_with_unified_api(self, query: str) -> dict:
        """
        使用UnifiedSearch API执行搜索（支持Tags和更多高级功能）

        Args:
            query: 搜索查询词

        Returns:
            dict: 搜索响应，包含Tags信息
        """
        logger.info(
            f"使用UnifiedSearch API搜索: query='{query}', engine_type={self.engine_type}"
        )

        # 构造 UnifiedSearch 请求
        unified_search_request = models.UnifiedSearchRequest(
            body=models.UnifiedSearchInput(
                query=query,
                engine_type=self.engine_type,
                time_range=self.time_range,
                contents=models.RequestContents(
                    main_text=True,  # 返回主要文本内容
                    summary=False,  # 不返回摘要（避免额外收费）
                    rerank_score=False,  # 不启用重排序（避免额外收费）
                ),
                advanced_params={"numResults": f"{self.search_count}"},
            )
        )

        try:
            start_time = time.time()
            response = await self.Client.unified_search_async(unified_search_request)
            search_time = time.time() - start_time

            request_id = response.body.request_id
            server_time = response.body.search_information.search_time
            result_count = len(response.body.page_items)

            logger.info(
                f"UnifiedSearch API成功: request_id={request_id}, "
                f"server_time={server_time}ms, total_time={search_time:.2f}s, "
                f"result_count={result_count}"
            )

            # 转换为字典格式便于处理
            result_dict = {
                "requestId": request_id,
                "pageItems": [],
                "searchInformation": {
                    "searchTime": server_time,
                    "total": getattr(response.body.search_information, "total", -1),
                },
            }

            # 转换 pageItems
            for item in response.body.page_items:
                page_item = {
                    "title": item.title or "",
                    "link": item.link or "",
                    "snippet": item.snippet or "",
                    "mainText": getattr(item, "main_text", "") or "",
                    "publishTime": getattr(item, "published_time", None),
                    "score": getattr(item, "score", None),
                }

                # 添加Tags信息（如果存在）
                if hasattr(item, "tags") and item.tags:
                    tags_dict = {}
                    if hasattr(item.tags, "genre"):
                        tags_dict["genre"] = item.tags.genre
                    if hasattr(item.tags, "is_ugc"):
                        tags_dict["isUgc"] = str(item.tags.is_ugc).lower()
                    if hasattr(item.tags, "ugc_type"):
                        tags_dict["ugcType"] = item.tags.ugc_type
                    if hasattr(item.tags, "industry"):
                        tags_dict["industry"] = item.tags.industry
                    if hasattr(item.tags, "is_list_page"):
                        tags_dict["isListPage"] = str(item.tags.is_list_page).lower()

                    if tags_dict:
                        page_item["tags"] = tags_dict
                        logger.debug(
                            f"提取Tags成功: title='{item.title}', tags={tags_dict}"
                        )

                result_dict["pageItems"].append(page_item)

            return result_dict

        except TeaException as e:
            code = e.code
            request_id = e.data.get("requestId") if hasattr(e, "data") else "N/A"
            message = e.data.get("message") if hasattr(e, "data") else str(e)
            logger.warning(
                f"UnifiedSearch API异常: "
                f"request_id={request_id}, code={code}, message={message}"
            )
            return None
        except Exception as e:
            logger.warning(f"UnifiedSearch API调用异常: {str(e)}")
            return None

    def _retrieve(self, query_bundle):
        """同步检索接口，不推荐使用，建议使用异步版本"""
        logger.warning("使用了同步检索接口，建议使用异步版本以获得更好的性能")
        import asyncio

        return asyncio.run(self._aretrieve(query_bundle))

    def set_engine_type(self, engine_type: str) -> None:
        """
        设置搜索引擎类型

        Args:
            engine_type: 搜索引擎类型 (Generic, GenericAdvanced, LiteAdvanced, Deep)
        """
        if engine_type not in SUPPORTED_ENGINE_TYPES:
            logger.warning(f"不支持的引擎类型: {engine_type}, 支持的类型: {SUPPORTED_ENGINE_TYPES}")
            return

        old_engine_type = self.engine_type
        self.engine_type = engine_type
        logger.info(f"搜索引擎类型已更改: {old_engine_type} -> {engine_type}")

    def get_engine_type(self) -> str:
        """获取当前搜索引擎类型"""
        return self.engine_type

    def enable_result_optimization(self, enable: bool = True) -> None:
        """
        启用/禁用搜索结果优化

        Args:
            enable: 是否启用优化
        """
        old_status = self.enable_optimization
        self.enable_optimization = enable
        logger.info(
            f"搜索结果优化: {'启用' if old_status else '禁用'} -> {'启用' if enable else '禁用'}"
        )

    def set_optimization_params(
        self, min_results: int = None, max_results: int = None
    ) -> None:
        """
        设置优化参数

        Args:
            min_results: 最少保留结果数
            max_results: 最多保留结果数
        """
        if min_results is not None:
            self.min_results = max(1, min_results)
            logger.info(f"最少保留结果数设置为: {self.min_results}")

        if max_results is not None:
            self.max_results = max(self.min_results, max_results)
            logger.info(f"最多保留结果数设置为: {self.max_results}")

    def get_config_info(self) -> Dict:
        """获取当前配置信息"""
        return {
            "engine_type": self.engine_type,
            "search_count": self.search_count,
            "time_range": self.time_range,
            "search_lang": self.search_lang,
            "enable_optimization": self.enable_optimization,
            "min_results": self.min_results,
            "max_results": self.max_results,
            "supported_engine_types": SUPPORTED_ENGINE_TYPES,
        }

    async def _aretrieve(
        self,
        query_bundle: QueryBundle,
    ):
        start = time.time()
        query_str = query_bundle.query_str

        logger.info(
            f"开始阿里云搜索: query='{query_str}', engine_type={self.engine_type}, "
            f"search_count={self.search_count}, optimization={'启用' if self.enable_optimization else '禁用'}"
        )

        search_results = await self._search_with_unified_api(query=query_str)

        # 合并所有搜索结果
        total_original_count = 0
        items = 0

        if search_results:
            items = search_results.get("pageItems", [])
            total_original_count += len(items)

            # 如果启用了优化且结果包含Tags信息，则进行优化
            if self.enable_optimization and any(item.get("tags") for item in items):
                logger.info(f"对搜索结果应用Tags优化: 原始数量={len(items)}")
                optimized_result = optimize_search_results(
                    search_results,
                    min_results=self.min_results,
                    max_results=self.max_results,
                )
                items = optimized_result.get("pageItems", [])

                # 记录优化统计
                optimization = optimized_result.get("optimization", {})
                logger.info(
                    f"搜索结果优化统计: 原始{optimization.get('original_count', 0)}条, "
                    f"过滤{optimization.get('filtered_count', 0)}条, "
                    f"排除{optimization.get('excluded_count', 0)}条, "
                    f"降级={'是' if optimization.get('degraded', False) else '否'}"
                )

        # 转换为 NodeWithScore
        nodes = []

        for item in items:
            # 提取文本内容
            text = ""
            if item.get("snippet"):
                text += item.get("snippet") + "\n\n"

            mainText = item.get("mainText") or item.get("markdownText")
            if mainText:
                text += mainText

            if not text:
                logger.debug(f"跳过无文本内容的结果: title='{item.get('title', 'N/A')}'")
                continue

            # 获取评分
            score = item.get("score", 0.1)
            if isinstance(score, str):
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 0.1

            # 创建节点
            node = TextNode(
                text=text[:800],  # 限制文本长度
                metadata={
                    "source": "web_search",
                    "file_url": item.get("link"),
                    "file_name": item.get("title") or item.get("htmlTitle"),
                    "host_name": item.get("hostname"),
                    "host_logo": item.get("hostLogo"),
                    "publish_time": item.get("publishTime"),
                    "search_engine": "aliyun",
                    "engine_type": self.engine_type,
                    "has_tags": bool(item.get("tags")),
                    "tags": item.get("tags", {}),
                },
            )

            nodes.append(NodeWithScore(node=node, score=score))

        elapsed_time = time.time() - start

        logger.info(
            f"阿里云搜索完成: 原始结果={total_original_count}条, "
            f"最终返回={len(nodes)}条, 耗时={elapsed_time:.2f}s, "
            f"engine_type={self.engine_type}, optimization={'启用' if self.enable_optimization else '禁用'}"
        )

        return nodes
