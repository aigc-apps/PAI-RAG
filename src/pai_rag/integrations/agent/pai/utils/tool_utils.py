from llama_index.core.tools import FunctionTool
from pai_rag.integrations.agent.pai.base_tool import ApiTool, PaiAgentDefinition
from pai_rag.integrations.agent.pai.utils.default_tool_description_template import (
    DEFAULT_CALCULATE_MULTIPLY,
    DEFAULT_CALCULATE_ADD,
    DEFAULT_CALCULATE_DIVIDE,
    DEFAULT_CALCULATE_SUBTRACT,
)
import logging

logger = logging.getLogger(__name__)


def get_calculator_tools():
    def multiply(a: int, b: int) -> int:
        """Multiply two integers and returns the result integer"""
        return a * b

    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer"""
        return a + b

    def divide(a: int, b: int) -> float:
        """Divide two integers and returns the result as a float"""
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    def subtract(a: int, b: int) -> int:
        """Subtract the second integer from the first and returns the result integer"""
        return a - b

    multiply_tool = FunctionTool.from_defaults(
        fn=multiply,
        name="calculate_multiply",
        description=DEFAULT_CALCULATE_MULTIPLY,
    )
    add_tool = FunctionTool.from_defaults(
        fn=add, name="calculate_add", description=DEFAULT_CALCULATE_ADD
    )
    divide_tool = FunctionTool.from_defaults(
        fn=divide, name="calculate_divide", description=DEFAULT_CALCULATE_DIVIDE
    )
    subtract_tool = FunctionTool.from_defaults(
        fn=subtract,
        name="calculate_subtract",
        description=DEFAULT_CALCULATE_SUBTRACT,
    )
    return [multiply_tool, add_tool, divide_tool, subtract_tool]


def generate_api_function(api_tool: ApiTool):
    # 获取API信息
    function_name = api_tool.name
    url = api_tool.url
    headers = api_tool.headers
    method = api_tool.method
    # GET -> params, POST -> json / data
    # 生成函数的参数
    required_params = api_tool.required
    param_str = ", ".join(required_params)

    if method.lower() == "get":
        param_name = "params"
    else:
        param_name = "data"
        headers["Content-Type"] = api_tool.content_type

    # 定义函数体
    function_body = f"""
def {function_name}({param_str}):
    url = '{url}'
    data = {{k: v for k, v in locals().items() if k in {list(required_params)}}}
    headers = {headers}
    method = '{method.lower()}'
    if method == 'post' and headers.get('Content-Type', 'application/json') == 'application/json':
        import json
        data = json.dumps(data)
    logger.info(f"Send requests to url {{url}}, data: {{data}}, headers: {{headers}}.")
    response = requests.{method.lower()}(url, {param_name}=data, headers=headers)

    if response.status_code == 200:
        logger.info(f"Send requests to url {{url}} success. Status code: {{response.status_code}}, response: {{response.text}}")
        return response.text
    else:
        logger.info(f"Send requests to url {{url}} failed. Status code: {{response.status_code}}, response: {{response.text}}")
        response.raise_for_status()
    """
    logger.info(function_body)

    # 将函数添加到当前模块的执行环境
    exec(function_body, globals())
    # 返回生成的函数
    return globals()[function_name]


def get_customized_tools(agent_definition: PaiAgentDefinition):
    tools = []

    # 首先尝试加载python代码
    if agent_definition.python_scripts:
        exec(agent_definition.python_scripts, globals())
        logger.info(f"Loaded python scripts: {agent_definition.python_scripts}")

    for api_tool in agent_definition.api_tools:
        api_func = generate_api_function(api_tool=api_tool)
        tool = FunctionTool.from_defaults(
            name=api_tool.name,
            description=api_tool.description,
            fn=api_func,
        )
        print(f"Loaded api tool definition {tool.metadata}")
        tools.append(tool)

    for func_tool in agent_definition.function_tools:
        tool = FunctionTool.from_defaults(
            name=func_tool.function.name,
            description=func_tool.function.description,
            fn=globals()[func_tool.function.name],
        )
        print(f"Loaded function tool definition {tool.metadata}")
        tools.append(tool)
    return tools
