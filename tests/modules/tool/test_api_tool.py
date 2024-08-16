from pai_rag.modules.tool.utils import get_customized_api_tools

config = {
    "functions": [
        {
            "name": "search_flight_ticket_api",
            "api": "http://127.0.0.1:8070/demo/api/flights",
            "headers": {"Authorization": "Bearer YOUR_ACCESS_TOKEN"},
            "method": "GET",
            "request_body_type": "params",
            "description": "帮助用户获取机票信息，用户需要输入出发地、目的地",
            "parameters": {
                "from_city": {"type": "str", "description": "出发城市，如'北京'、'上海'、'南京''"},
                "to_city": {"type": "str", "description": "目的地城市，如'北京'、'上海'、'南京'"},
                "date": {"type": "str", "description": "出发时间，如'2024-03-29'"},
            },
            "required": ["from_city", "to_city", "date"],
        }
    ]
}


def test_customized_api_tool():
    tools = get_customized_api_tools(config)
    assert len(tools) == 1
    assert tools[0].metadata.name == "search_flight_ticket_api"
    assert tools[0].metadata.description == "帮助用户获取机票信息，用户需要输入出发地、目的地"
    assert tools[0].fn.__name__ == "search_flight_ticket_api"
