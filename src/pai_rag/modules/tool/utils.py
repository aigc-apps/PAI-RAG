from llama_index.tools.google import GoogleSearchToolSpec
from pai_rag.modules.tool.load_and_search_tool_spec import LoadAndSearchToolSpec
from llama_index.core.tools import FunctionTool
from llama_index.core.bridge.pydantic import FieldInfo, create_model
import requests
from pai_rag.modules.tool.default_tool_description_template import (
    DEFAULT_GOOGLE_SEARCH_TOOL_DESP,
    DEFAULT_CALCULATE_MULTIPLY,
    DEFAULT_CALCULATE_ADD,
    DEFAULT_CALCULATE_DIVIDE,
    DEFAULT_CALCULATE_SUBTRACT,
    DEFAULT_GET_WEATHER,
)


def create_tool_fn_schema(name, params):
    fields = {}
    params_prop = params["properties"]
    for param_name in params_prop:
        param_type = params_prop[param_name]["type"]
        param_desc = params_prop[param_name]["description"]
        fields[param_name] = (param_type, FieldInfo(description=param_desc))
    return create_model(name, **fields)  # type: ignore


def create_custom_tool_fn_schema(name, params_prop):
    fields = {}
    for param_name in params_prop:
        param_type = params_prop[param_name]["type"]
        param_desc = params_prop[param_name]["description"]
        fields[param_name] = (param_type, FieldInfo(description=param_desc))
    return create_model(name, **fields)  # type: ignore


def get_google_web_search_tools(config):
    google_spec = GoogleSearchToolSpec(
        key=config.get("google_search_api", None),
        engine=config.get("google_search_engine", None),
        num=10,
    )
    return LoadAndSearchToolSpec.from_defaults(
        tool=google_spec.to_tool_list()[0],
        name="google_search",
        description=DEFAULT_GOOGLE_SEARCH_TOOL_DESP,
    ).to_tool_list()


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


def get_weather_tools(config):
    def get_place_weather(city: str) -> str:
        """Get city name and return city weather"""
        api_key = config.get("weather_api_key", None)
        # 可以直接赋值给api_key,原始代码的config只有type类型。
        base_url = "http://api.openweathermap.org/data/2.5/forecast?"
        complete_url = f"{base_url}q={city}&appid={api_key}&lang=zh_cn&units=metric"
        response = requests.get(complete_url)
        weather_data = response.json()

        if weather_data["cod"] != "200":
            print(f"获取天气信息失败，错误代码：{weather_data['cod']}")
            return None

        element = weather_data["list"][0]

        return str(
            f"{city}的天气:\n 时间: {element['dt_txt']}\n 温度: {element['main']['temp']} °C\n 天气描述: {element['weather'][0]['description']}\n"
        )

    weather_tool = FunctionTool.from_defaults(
        fn=get_place_weather,
        name="get_weather",
        description=DEFAULT_GET_WEATHER,
    )

    return [weather_tool]


def get_customized_api_tools(config):
    def generate_api_function(api_info):
        # 获取API信息
        function_name = api_info["name"]
        api_url = api_info["api"]
        headers = api_info["headers"]
        method = api_info["method"]
        request_body_type = api_info["request_body_type"]
        # GET -> params, POST -> json / data
        # 生成函数的参数
        required_params = api_info["required"]
        param_str = ", ".join(required_params)

        # 定义函数体
        function_body = f"""
def {function_name}({param_str}):
    url = '{api_url}'
    data = {{k: v for k, v in locals().items() if k in {list(required_params)}}}
    headers = {headers}
    response = requests.{method.lower()}(url, {request_body_type}=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()
"""

        # 将函数添加到当前模块的执行环境
        exec(function_body, globals())
        # 返回生成的函数
        return globals()[function_name]

    tools = []
    for api_info in config["functions"]:
        # 生成API函数
        func = generate_api_function(api_info)
        fn_name = api_info["name"]
        fn_schema = create_custom_tool_fn_schema(fn_name, api_info["parameters"])
        tool = FunctionTool.from_defaults(
            fn=func,
            name=fn_name,
            fn_schema=fn_schema,
            description=api_info["description"],
        )
        tools.append(tool)
    return tools


def get_customized_python_tools(config, function_body_str):
    def generate_python_functions(config, function_body_str):
        # 将函数添加到当前模块的执行环境
        exec(function_body_str, globals())
        func_dict = {}
        for func_info in config["functions"]:
            function_name = func_info["name"]
            func = func_info["function"]
            func_dict[function_name] = globals()[func]
        return func_dict

    tools = []
    func_dict = generate_python_functions(config, function_body_str)
    for func_info in config["functions"]:
        fn_name = func_info["name"]
        func = func_dict[fn_name]
        fn_schema = create_custom_tool_fn_schema(fn_name, func_info["parameters"])
        tool = FunctionTool.from_defaults(
            fn=func,
            name=fn_name,
            fn_schema=fn_schema,
            description=func_info["description"],
        )
        tools.append(tool)
    return tools
