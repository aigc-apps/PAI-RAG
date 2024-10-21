import tempfile
import json
from pai_rag.integrations.agent.pai.pai_agent import AgentConfig
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
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Write some data to the temporary file
        function_json_data = json.dumps(function_json).encode("utf-8")
        temp_file.write(function_json_data)
        temp_file.flush()
        temp_filename = temp_file.name  # Get the name of the temp file
        agent_config = AgentConfig(tool_definition_file=temp_filename)
        tools = get_customized_tools(agent_config)
        assert tools[0].metadata.name == "search_flight_ticket_api"
        assert tools[0].metadata.description == "帮助用户获取机票信息，用户需要输入出发地、目的地"
        assert tools[0].fn.__name__ == "search_flight_ticket_api"
