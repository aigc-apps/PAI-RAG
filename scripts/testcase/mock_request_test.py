import requests
import json
import time

# 配置请求参数
url = "http://localhost:8087/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = [
    {
        "model": "DeepSeek-R1",  #  模型名称
        "stream": False,
        "chat_news": True,
        "search_web": True,
        "messages": [{"role": "user", "content": "下一届奥运会是什么时候"}],
    }
]

results = []  # 存储结构化结果

for i in range(10):  # 循环发送 len(data) 次请求
    print(f"发送第 {i} 次请求...")
    try:
        response = requests.post(url, headers=headers, json=data[0])  # 发送请求
        response_json = response.json()

        # 提取关键信息
        output_content = response_json["choices"][0]["message"][
            "content"
        ]  # 仅提取 message.content

        # 构建结构化数据
        results.append({"input": data[0], "output": output_content})

        time.sleep(1)  # 避免请求过快

    except Exception as e:
        print(f"请求失败: {str(e)}")

# 写入结构化 JSON 文件
with open("output_web_deepseek_r1.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)  # 格式化保存

print(f"已完成{len(data)}次请求，结果已保存。")
