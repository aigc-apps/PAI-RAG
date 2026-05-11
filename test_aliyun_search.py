#!/usr/bin/env python3
"""
阿里云搜索测试脚本
使用环境变量 ALIYUN_ACCESS_KEY_ID 和 ALIYUN_ACCESS_KEY_SECRET 进行认证
"""

import os
import sys
import asyncio
import json
import random
from typing import Dict
from dotenv import load_dotenv

from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_iqs20241111 import models
from alibabacloud_iqs20241111.client import Client
import time

load_dotenv()


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


def optimize_search_results(
    results: Dict, min_results: int = 5, max_results: int = 20
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
        if len(filtered_items) > max_results:
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
        filtered_items = page_items
        excluded_items = []

    # 构建优化后的结果
    optimized_results = results.copy()
    optimized_results["pageItems"] = filtered_items

    # 添加优化统计信息
    optimized_results["optimization"] = {
        "original_count": len(page_items),
        "filtered_count": len(filtered_items),
        "excluded_count": len(excluded_items),
        "degraded": degraded,
        "excluded_items": excluded_items[:3] if excluded_items else [],  # 只显示前3个排除项
    }

    return optimized_results


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


def simulate_tags_for_demo(results: Dict) -> Dict:
    """
    为演示目的模拟添加Tags数据
    基于阿里云搜索API文档的真实枚举值
    实际使用时，Tags数据由阿里云搜索API直接返回
    """
    simulated_results = results.copy()
    page_items = simulated_results.get("pageItems", [])

    # 基于API文档的真实Tags样本（用于演示不同的过滤效果）
    sample_tags = [
        # 高质量内容 - 应该保留
        {
            "genre": "NewsPortal",
            "isUgc": "false",
            "industry": "News",
            "isListPage": "false",
        },
        {
            "genre": "Encyclopedia",
            "isUgc": "false",
            "industry": "General",
            "isListPage": "false",
        },
        {
            "genre": "Blog",
            "isUgc": "true",
            "ugcType": "MediaArticle",
            "industry": "General",
            "isListPage": "false",
        },
        {
            "genre": "VideoSite",
            "isUgc": "true",
            "ugcType": "MediaArticle",
            "industry": "Entertainment",
            "isListPage": "false",
        },
        {
            "genre": "NewsPortal",
            "isUgc": "false",
            "industry": "Finance",
            "isListPage": "false",
        },
        # 低质量内容 - 应该被过滤
        {
            "genre": "ForumUgc",
            "isUgc": "true",
            "ugcType": "StructuredQA",
            "industry": "Auto",
            "isListPage": "false",
        },  # 问答类UGC
        {
            "genre": "Social",
            "isUgc": "true",
            "ugcType": "NoteShare",
            "industry": "General",
            "isListPage": "false",
        },  # 笔记分享
        {
            "genre": "Commerce",
            "isUgc": "false",
            "industry": "General",
            "isListPage": "true",
        },  # 列表页
        {
            "genre": "ForumUgc",
            "isUgc": "true",
            "ugcType": "ForumPost",
            "industry": "Tech",
            "isListPage": "false",
        },  # 论坛帖子
        {
            "genre": "Blog",
            "isUgc": "true",
            "ugcType": "StructuredQA",
            "industry": "Auto",
            "isListPage": "false",
        },  # 问答类博客
    ]

    for i, item in enumerate(page_items):
        # 为每个结果分配相应的tags样本
        if i < len(sample_tags):
            item["tags"] = sample_tags[i]
        else:
            # 超出样本数量时随机选择
            item["tags"] = random.choice(sample_tags)

    return simulated_results


class AliyunSearchTester:
    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        endpoint: str = "iqs.cn-hangzhou.aliyuncs.com",
        time_range: str = "OneMonth",
        use_unified_api: bool = True,
    ):
        """
        初始化阿里云搜索测试器

        Args:
            access_key_id: 阿里云 Access Key ID
            access_key_secret: 阿里云 Access Key Secret
            endpoint: 搜索服务端点
            time_range: 时间范围 (OneMonth, OneWeek, OneDay, OneYear, NoLimit)
            use_unified_api: 是否使用UnifiedSearch API（支持Tags）
        """
        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
        )
        config.endpoint = endpoint

        self.client = Client(config)
        self.time_range = time_range
        self.use_unified_api = use_unified_api
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = endpoint

    async def search_with_unified_api(self, query: str) -> dict:
        """
        使用UnifiedSearch API执行搜索（支持Tags和更多高级功能）

        Args:
            query: 搜索查询词

        Returns:
            dict: 搜索响应，包含Tags信息
        """
        from Tea.exceptions import TeaException

        print(f"🔍 搜索查询: {query}")
        print("🌐 使用 UnifiedSearch API")
        print(f"📍 端点: {self.endpoint}")
        print("-" * 50)

        # 构造 UnifiedSearch 请求
        unified_search_request = models.UnifiedSearchRequest(
            body=models.UnifiedSearchInput(
                query=query,
                engine_type="LiteAdvanced",
                time_range=self.time_range,  # NoLimit, OneMonth, OneWeek, OneDay, OneYear
                contents=models.RequestContents(
                    main_text=True,  # 返回主要文本内容
                    summary=False,  # 不返回摘要（避免额外收费）
                    rerank_score=False,  # 不启用重排序（避免额外收费）
                ),
            )
        )

        try:
            start_time = time.time()
            response = await self.client.unified_search_async(unified_search_request)
            search_time = time.time() - start_time

            # 打印响应基本信息
            print("📊 请求成功")
            print(f"📨 请求ID: {response.body.request_id}")
            print(f"⏱️ 搜索耗时: {response.body.search_information.search_time}ms (服务端)")
            print(f"🕐 总耗时: {search_time:.2f}s")
            print(f"📈 结果数量: {len(response.body.page_items)}")

            # 转换为字典格式便于处理
            result_dict = {
                "requestId": response.body.request_id,
                "pageItems": [],
                "searchInformation": {
                    "searchTime": response.body.search_information.search_time,
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

                result_dict["pageItems"].append(page_item)

            print("✅ UnifiedSearch 搜索成功")
            return result_dict

        except TeaException as e:
            code = e.code
            request_id = e.data.get("requestId") if hasattr(e, "data") else "N/A"
            message = e.data.get("message") if hasattr(e, "data") else str(e)
            print(
                f"❌ UnifiedSearch API异常: requestId:{request_id}, code:{code}, message:{message}"
            )
            print("🔄 降级使用 GenericSearch API")
            return await self.search_with_generic_api(query)
        except Exception as e:
            print(f"❌ UnifiedSearch API 调用异常: {str(e)}")
            print("🔄 降级使用 GenericSearch API")
            return await self.search_with_generic_api(query)

    async def search_with_generic_api(self, query: str, page: int = 1) -> dict:
        """
        使用GenericSearch API执行搜索（兼容模式）

        Args:
            query: 搜索查询词
            page: 页码，从1开始

        Returns:
            dict: 搜索响应
        """
        request = models.GenericSearchRequest(
            query=query,
            time_range=self.time_range,
            page=page,
        )

        print(f"🔍 搜索查询: {query}")
        print(f"📄 页码: {page}")
        print(f"⏰ 时间范围: {self.time_range}")
        print(f"🌐 端点: {self.client._endpoint}")
        print("-" * 50)

        try:
            response = await self.client.generic_search_async(request)

            # 打印响应基本信息
            print(f"📊 状态码: {response.status_code}")
            print(f"📨 请求ID: {response.headers.get('x-acs-request-id', 'N/A')}")

            if response.status_code == 200:
                response_dict = response.body.to_map()
                print("✅ 搜索成功")
                print(f"📈 结果数量: {len(response_dict.get('pageItems', []))}")
                return response_dict
            else:
                print(f"❌ 搜索失败: {response.status_code}")
                print(f"错误详情: {response}")
                return {}

        except Exception as e:
            print(f"❌ 搜索异常: {str(e)}")
            return {}

    async def search(self, query: str, page: int = 1) -> dict:
        """
        执行搜索并返回原始响应

        Args:
            query: 搜索查询词
            page: 页码，从1开始

        Returns:
            dict: 原始搜索响应
        """
        if self.use_unified_api:
            return await self.search_with_unified_api(query)
        else:
            return await self.search_with_generic_api(query, page)

    def print_search_results(self, results: dict):
        """
        格式化打印搜索结果

        Args:
            results: 搜索结果字典
        """
        print("\n" + "=" * 80)
        print("📋 搜索结果详情")
        print("=" * 80)

        # 打印完整的原始响应（格式化JSON），排除优化统计信息
        print("\n🔧 原始响应:")
        display_results = {k: v for k, v in results.items() if k != "optimization"}
        print(json.dumps(display_results, indent=2, ensure_ascii=False))

        # 解析并显示搜索项目
        page_items = results.get("pageItems", [])
        if page_items:
            print(f"\n📑 解析后的搜索项目 (共 {len(page_items)} 条):")
            print("-" * 80)

            for i, item in enumerate(page_items, 1):
                print(f"\n[{i}] {item.get('title', 'N/A')}")
                print(f"🔗 链接: {item.get('link', 'N/A')}")
                print(f"🏠 主机: {item.get('hostname', 'N/A')}")
                print(f"📅 发布时间: {item.get('publishTime', 'N/A')}")
                print(f"⭐ 评分: {item.get('score', 'N/A')}")

                # 显示内容摘要
                main_text = item.get("mainText", "")
                markdown_text = item.get("markdownText", "")
                html_snippet = item.get("htmlSnippet", "")

                content = main_text or markdown_text or html_snippet
                if content:
                    # 截取前200个字符作为摘要
                    summary = content[:200] + "..." if len(content) > 200 else content
                    print(f"📝 内容摘要: {summary}")

                # 显示Tags信息（如果存在）
                tags = item.get("tags", {})
                if tags:
                    print(f"🏷️ Tags: {tags}")

                print("-" * 40)
        else:
            print("\n❌ 没有找到搜索结果")


async def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("使用方法: python test_aliyun_search.py <搜索查询>")
        print("例如: python test_aliyun_search.py '人工智能最新进展'")
        sys.exit(1)

    query = sys.argv[1]

    # 获取环境变量
    access_key_id = os.getenv("ALIYUN_ACCESS_KEY_ID")
    access_key_secret = os.getenv("ALIYUN_ACCESS_KEY_SECRET")

    if not access_key_id or not access_key_secret:
        print("❌ 错误: 请设置环境变量 ALIYUN_ACCESS_KEY_ID 和 ALIYUN_ACCESS_KEY_SECRET")
        print("设置方法:")
        print("export ALIYUN_ACCESS_KEY_ID='your_access_key_id'")
        print("export ALIYUN_ACCESS_KEY_SECRET='your_access_key_secret'")
        sys.exit(1)

    # 初始化搜索器
    tester = AliyunSearchTester(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        endpoint="iqs.cn-zhangjiakou.aliyuncs.com",  # 使用正确的端点
        use_unified_api=True,  # 使用支持Tags的UnifiedSearch API
    )

    # 执行搜索
    results = await tester.search(query)

    # 打印结果
    if results:
        print("\n" + "=" * 80)
        print("🔍 原始搜索结果")
        print("=" * 80)
        tester.print_search_results(results)

        print("\n" + "=" * 80)
        print("🚀 搜索结果优化演示")
        print("=" * 80)

        # 为演示目的添加模拟Tags数据
        print("📝 步骤1: 模拟添加Tags数据（实际使用时由API直接返回）")
        results_with_tags = simulate_tags_for_demo(results)

        # 应用搜索结果优化
        print("⚡ 步骤2: 应用基于Tags的搜索结果优化")
        optimized_results = optimize_search_results(
            results_with_tags, min_results=3, max_results=20
        )

        # 打印优化统计信息
        optimization = optimized_results.get("optimization", {})
        print("\n📊 优化统计:")
        print(f"   原始结果数量: {optimization.get('original_count', 0)}")
        print(f"   过滤后数量: {optimization.get('filtered_count', 0)}")
        print(f"   排除数量: {optimization.get('excluded_count', 0)}")
        print(f"   是否降级: {'是' if optimization.get('degraded', False) else '否'}")

        # 显示被排除的项目
        excluded_items = optimization.get("excluded_items", [])
        if excluded_items:
            print(f"\n🚫 被排除的结果示例（前{len(excluded_items)}项）:")
            for i, item in enumerate(excluded_items, 1):
                print(f"   [{i}] {item['title'][:50]}...")
                print(f"       排除原因: {item['exclude_reason']}")
                print(f"       Tags: {item['tags']}")
                print()

        # 打印优化后的结果
        print("\n" + "=" * 80)
        print("✨ 优化后的搜索结果")
        print("=" * 80)
        tester.print_search_results(optimized_results)

    else:
        print("❌ 搜索失败或无结果")


if __name__ == "__main__":
    asyncio.run(main())
