from asyncio.log import logger
from typing import List, Dict, Any
from datetime import datetime
import time
import hmac
import base64
from hashlib import sha256
import httpx
from urllib.parse import urljoin
import random
import string
import json


def random_string(n: int) -> str:
    return bytes(random.choices(string.ascii_lowercase.encode("ascii"), k=n)).decode(
        "ascii"
    )


def calc_signature(user: str, ts: str, salt: str, secret: str):
    """Generate signature"""
    data = f"{user}{ts}{salt}{secret}".encode("utf-8")
    signature = base64.b64encode(
        hmac.new(secret.encode("utf-8"), data, digestmod=sha256).digest()
    )
    return signature


async def get_access_token(host: str, user: str, secret: str):
    """Get access token"""
    salt = random_string(6)
    ts = str(int(round(time.time() * 1000)))
    sign = calc_signature(user, ts, salt, secret)
    logger.info(f"Quark login with user {user}, ts {ts}, salt {salt}, secret {secret}.")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            urljoin(host, "/api/auth/token"),
            json={
                "salt": salt,
                "timestamp": ts,
                "userName": user,
                "sign": sign.decode("utf-8"),
            },
        )
        response.raise_for_status()
        data = response.json()
        if "result" not in data or data.get("message") != "success":
            raise ValueError(f"Invalid quark login response: {data}")
        return data["result"]["token"], data["result"]["expireTime"]


def format_timestamp(ts):
    dt_object = datetime.fromtimestamp(ts)
    return dt_object.strftime("%Y-%m-%d %H:%M:%S")


def process_baike_item(item):
    result = {
        "url": "",
        "text": "",
    }

    if "baike_sc" in item:
        data = item["baike_sc"]
        result["url"] = data["url"]
        result["text"] = f'{data["sub_title"]}\n\n{data["abstract"]}\n\n'
        result["title"] = data["sub_title"]

    if "moduleData" in item:
        data = item["moduleData"]
        result[
            "text"
        ] += f'{data["title_text"]}\n{data["sub_title"]}\n\n{data["baike_info"]}\n\n'
        result["url"] = data["baike_url"]
        result["title"] = data["title_text"]

    return [result]


def process_news(item):
    merge_news_result = {"url": "", "title": "资讯新闻", "text": ""}
    for news_info in item["news_uchq"]["news_node"]:
        merge_news_result[
            "text"
        ] += f"""
            标题: {news_info["title"]}
            文本: {news_info["summary"]} {news_info["desc_text"]}
            时间: {news_info["time"]}
            链接: {news_info["url"]}
        """
    return [merge_news_result]


def process_wenda(item):
    results = []
    wenda_info = item["wenda_selected"]
    results.append(
        {
            "url": wenda_info["url"],
            "text": f'{wenda_info["name"]}\n\n{wenda_info["content"]}\n\n',
            "title": wenda_info["name"],
        }
    )

    return results


def process_weather(item):
    results = []
    weather_info = item["weather_moji"]["wisdomData"]
    results.append(
        {
            "url": "",
            "title": "墨迹天气",
            "text": f"{json.dumps(weather_info, ensure_ascii=False)}",
        }
    )
    return results


def process_weibo(item):
    merge_weibo_result = {"url": "", "title": "微博", "text": ""}
    for weibo_info in item["weibo_strong"]["list"]:
        merge_weibo_result[
            "text"
        ] += f"""
            用户: {weibo_info["name"]}
            文本: {weibo_info["text"]}
            时间: {weibo_info["time"]}
            链接: {weibo_info["url"]}

        """
    return [merge_weibo_result]


def process_structured_web(item):
    results = []
    results.append(
        {
            "url": item["url"],
            "title": item["article_title"],
            "text": f'{item["title"]}\n\n{item["MainBody"]}',
            "time": item["time"],
        }
    )
    return results


def process_top(item):
    merge_news_result = {"url": "", "title": "新闻热榜", "text": ""}
    for top_info in item["news_top_list"]["display"]["fields"]:
        merge_news_result[
            "text"
        ] += f"""
            标题: {top_info["news_title"]}
            文本: {top_info["news_summary"]}
            时间: {top_info["publish_time"]}
            链接: {top_info["url"]}

        """
    return [merge_news_result]


def postprocess_items(items: List[Dict[str, Any]]):
    results = []
    for item in items:
        if (
            "sc_name" not in item
            and "url" in item
            and "title" in item
            and "MainBody" in item
        ):
            results.append(
                {
                    "url": item["url"],
                    "title": item["title"],
                    "text": f'{item["title"]}\n\n{item["MainBody"]}\n\n',
                }
            )
        elif item["sc_name"] == "news_top_list":
            results.extend(process_top(item))
        elif item["sc_name"] == "weibo_strong":
            results.extend(process_weibo(item))
        elif item["sc_name"] == "weather_moji":
            results.extend(process_weather(item))
        elif item["sc_name"] == "wenda_selected":
            results.extend(process_wenda(item))
        elif item["sc_name"] == "structure_web_info":
            results.extend(process_structured_web(item))
        elif item["sc_name"] == "news_uchq":
            results.extend(process_news(item))
        elif item["sc_name"] == "baike_sc":
            results.extend(process_baike_item(item))
    return results
