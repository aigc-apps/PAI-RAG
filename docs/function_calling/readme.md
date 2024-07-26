<p align="center">
    <h1>Function Calling in PAI-RAG </h1>
</p>

# 什么是 Function Calling

OpenAI于23年6月份的更新的 gpt-4-0613 和 gpt-3.5-turbo-0613 版本中为模型添加了Function Calling功能，通过给模型提供一组预定义的函数（Function list）以及用户提问（Query），让大模型自主选择要调用的函数，并向该函数提供所需的输入参数。随后我们就可以在自己的环境中基于模型生成的参数调用该函数，并将结果返回给大模型。

ChatGPT的Function Calling功能在发布之后立刻引起了人们的关注，因其简单易用的特性以及规范的输入输出迅速成为模型生态中Function calling的格式规范。后续的具有function calling功能的模型有很多参照了OpenAI的Function Calling格式，其输入的函数列表以及输出的Function Calling及其参数都以JSON格式输出：输入的函数列表中通常包括函数名称、函数功能描述、函数参数等部分，而输出中则按顺序输出所调用的函数名称和其中使用的参数。

# 如何定义 Function 及其实现

在PAI-RAG中，支持用户调用自定义的functions与外部世界进行更多的交互，如对接会议室预定系统，可以查询预定状态，协助用户提供个人信息来预定所需会议室；如对接数据库系统，调用外部API等。

PAI-RAG中的function定义遵循OpenAI的Function Calling格式规范，如下定义了在会议室预定系统中如何查询预订状态的function。

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_booking_state",
      "description": "用于查询给定会议室ID的预定状态",
      "parameters": {
        "type": "object",
        "properties": {
          "room_id": {
            "type": "str",
            "description": "待查询的会议室ID"
          }
        },
        "required": ["room_id"]
      }
    }
  }
]
```

定义好function之后，需要将对应function name的逻辑进行python实现，如下：

```python
def get_booking_state(room_id: str) -> str:
    try:
        return str(bookings[room_id].dict())
    except Exception:
        return f"没有找到会议室ID:{room_id}"
```

按照如上规范定义好所需的function即可在PAI-RAG中进行调用。

# PAI-RAG如何调用demo functions

我们提供了一组预置的会议室预定系统的demo，在 src/pai_rag/modules/tool/booking_demo 目录下，可以直接以demo启动体验：

```bash
pai_rag serve -p 8071 -c src/pai_rag/config/settings_fc_demo.toml
```

[注意] 在demo中，我们默认使用qwen2-7b-instruct作为function calling的LLM，您也可以根据需要替换为更大的模型，如qwen2-72b-instruct。

1. 发送预定会议室的请求

```bash
# Query
curl -X 'POST' http://127.0.0.1:8071/service/query/agent -H "Content-Type: application/json" -d '{"question":"我要预定303会议室"}'

# Response
Added user message to memory: 我要预定303会议室

=== Calling Function ===
Calling function: create_booking with args: {"room_id": "303"}
=== Function Output ===
会议室预订ID:303已创建，但尚未确认。请提供您的姓名(name)、电子邮件(email)、电话(phone)、日期(date)和时间(time)。

=== LLM Response ===
会议室303的预定已经创建成功，但还需要进一步确认。请您提供以下信息以便完成预定：

- 姓名（name）
- 电子邮件（email）
- 电话（phone）
- 日期（date）
- 时间（time）

请将这些信息告诉我，我将为您更新预定详情。
```

2. 完善用户的预定信息

```bash
# Query
curl -X 'POST' http://127.0.0.1:8071/service/query/agent -H "Content-Type: application/json" -d '{"question":"我叫玛丽，手机号是1505185"}'

# Response
Added user message to memory: 我叫玛丽，手机号是1505185

=== Calling Function ===
Calling function: update_booking with args: {"room_id": "303", "property": "name", "value": "\u739b\u4e3d"}
=== Function Output ===
预订ID 303 更新属性信息 name = 玛丽
=== Calling Function ===
Calling function: update_booking with args: {"room_id": "303", "property": "phone", "value": "1505185"}
=== Function Output ===
预订ID 303 更新属性信息 phone = 1505185
=== LLM Response ===
您的名字和电话号码已更新：
- 姓名：玛丽
- 电话：1505185

还请提供您的电子邮件地址、预定的日期和时间，以便完成会议室预定的所有必要信息。
```

```bash
# Query
curl -X 'POST' http://127.0.0.1:8071/service/query/agent -H "Content-Type: application/json" -d '{"question":"电子邮件是123@qq.com，预定日期是2024年8月10号，时间是下午6点"}'

# Response
Added user message to memory: 电子邮件是123@qq.com，预定日期是2024年8月10号，时间是下午6点

=== Calling Function ===
Calling function: update_booking with args: {"room_id": "303", "property": "email", "value": "123@qq.com"}
=== Function Output ===
预订ID 303 更新属性信息 email = 123@qq.com
=== Calling Function ===
Calling function: update_booking with args: {"room_id": "303", "property": "date", "value": "2024-08-10"}
=== Function Output ===
预订ID 303 更新属性信息 date = 2024-08-10
=== Calling Function ===
Calling function: update_booking with args: {"room_id": "303", "property": "time", "value": "18:00"}
=== Function Output ===
预订ID 303 更新属性信息 time = 18:00

=== LLM Response ===
您的预定信息已经全部更新完毕：
- 姓名：玛丽
- 电话：1505185
- 电子邮件：123@qq.com
- 日期：2024年8月10日
- 时间：下午6点（18:00）

会议室预定成功。如果您需要进一步的帮助或有其他要求，请随时告诉我。
```

3. 查询会议室预定状态

```bash
# Query
curl -X 'POST' http://127.0.0.1:8071/service/query/agent -H "Content-Type: application/json" -d '{"question":"查询303有没有被预定"}'

# Response
Added user message to memory: 查询303有没有被预定

=== Calling Function ===
Calling function: get_booking_state with args: {"room_id": "303"}
=== Function Output ===
{'name': '玛丽', 'email': '123@qq.com', 'phone': '1505185', 'date': '2024-08-10', 'time': '18:00'}

=== LLM Response ===
会议室303已被预定，预定信息如下：
- 姓名：玛丽
- 电子邮件：123@qq.com
- 电话：1505185
- 日期：2024年8月10日
- 时间：下午6点（18:00）

如果您需要预定该会议室或其他帮助，请告知我。
```

按照如上所做可以看到我们已经和会议室预定系统建立了连接，并且能够有效且快速处理用户的请求，且工具调用的准确率很高。

# PAI-RAG如何调用用户自定义的 functions

除了我们预置的demo，也支持用户实现任意自己的function函数：

1. 在 src/pai_rag/modules/tool 目录下新建一个自定义的文件夹，如 user_tool
2. 在 src/pai_rag/modules/tool/user_tool目录下新建两个文件，分别为：custom_functions.json 和 custom_functions.py。并分别按照前面所述实现自己的函数定义和实现逻辑即可。
3. 修改 src/pai_rag/config/settings_fc_demo.toml 文件的最后一行，将 func_path = "src/pai_rag/modules/tool/booking_demo" 替换为 第一步中的目录名称（src/pai_rag/modules/tool/user_tool）即可。
4. 指定配置文件来启动项目：pai_rag serve -p 8071 -c src/pai_rag/config/settings_fc_demo.toml

之后，您就可以进行请求调用来查看function calling的效果。
