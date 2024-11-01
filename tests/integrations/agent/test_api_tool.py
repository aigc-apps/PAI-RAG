from pai_rag.integrations.agent.pai.base_tool import PaiAgentDefinition
from pai_rag.integrations.agent.pai.utils.tool_utils import get_customized_tools

function_json = {
    "api_tools": [
        {
            "name": "search_flight_ticket_api",
            "url": "http://127.0.0.1:8070/demo/api/flights",
            "headers": {"Authorization": "Bearer YOUR_ACCESS_TOKEN"},
            "method": "GET",
            "content_type": "params",
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
    json_object = {
        "system_prompt": "",
        "api_tools": function_json["api_tools"],
        "python_scripts": "",
        "function_tools": [],
    }

    agent_definition = PaiAgentDefinition.model_validate(json_object)

    tools = get_customized_tools(agent_definition)
    assert tools[0].metadata.name == "get_current_datetime"

    assert tools[1].metadata.name == "search_flight_ticket_api"
    assert tools[1].metadata.description == "帮助用户获取机票信息，用户需要输入出发地、目的地"
    assert tools[1].fn.__name__ == "search_flight_ticket_api"
