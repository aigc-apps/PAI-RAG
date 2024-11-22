import requests
import os
from loguru import logger


def get_place_weather(city: str) -> str:
    logger.info(f"[Agent] Checking realtime weather info for {city}")

    """Get city name and return city weather"""
    api_key = os.environ.get("weather_api_key")

    # 可以直接赋值给api_key,原始代码的config只有type类型。
    base_url = "http://api.openweathermap.org/data/2.5/forecast?"
    complete_url = f"{base_url}q={city}&appid={api_key}&lang=zh_cn&units=metric"
    logger.info(f"Requesting {complete_url}...")
    response = requests.get(complete_url, timeout=5)
    weather_data = response.json()

    if weather_data["cod"] != "200":
        logger.error(
            f"获取天气信息失败，错误代码：{weather_data['cod']} 错误信息：{weather_data['message']}"
        )
        return f"获取天气信息失败，错误代码：{weather_data['cod']} 错误信息：{weather_data['message']}"

    element = weather_data["list"][0]

    return f"""
        {city}的天气:
        时间: {element['dt_txt']}
        温度: {element['main']['temp']} °C
        天气描述: {element['weather'][0]['description']}
    """
