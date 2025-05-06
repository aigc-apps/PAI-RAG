import os
import json
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


origins = [
    "*"
    # "http://localhost:3000",  # 前端开发服务器地址
    # "http://127.0.0.1:3000",
    # # 添加其他需要允许的域名
]

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有的HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)


# 数据模型
class MCPConfig(BaseModel):
    id: int
    name: str
    url: str
    type: str
    active: bool

class LLMConfig(BaseModel):
    id: int
    source: str
    model_name: str
    api_key: str
    max_context: int

class ConfigRequest(BaseModel):
    llm_config: Optional[List[LLMConfig]] = None
    mcp_config: Optional[List[MCPConfig]] = None



CONFIG_FILE = "config.json"

# 自动创建默认配置文件
if not os.path.exists(CONFIG_FILE):
    default_config = {
        "llm_config":  [],
        "mcp_config": []
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(default_config, f, indent=2)


@app.get("/api/configs", response_model=dict)
def read_configs():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取配置失败: {str(e)}")

@app.get("/api/configs/models", response_model=dict)
def get_models():
    try:
        with open(CONFIG_FILE, "r") as f:
            data =  json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取配置失败: {str(e)}")
    
    grouped = {}

    # 处理 llm_config
    for config in data.get("llm_config", []):
        source = config.get("source")
        model_name = config.get("model_name")
        if not source or not model_name:
            continue  # 跳过无效配置

        if source not in grouped:
            grouped[source] = []

        grouped[source].append({"name": model_name, "api_key": config.get("api_key")})
        
    return {
        "groups": [
            {
                "id": source.lower(),
                "label": source,
                "models": models
            }
            for source, models in grouped.items()
        ]
    }
    # return {
    #     "groups": [
    #         {
    #         "id": "openai",
    #         "label": "OpenAI",
    #         "models": [
    #             { "name": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo" },
    #             { "name": "gpt-4", "label": "GPT-4" }
    #         ]
    #         },
    #         {
    #         "id": "qwen",
    #         "label": "Qwen",
    #         "models": [
    #             { "name": "qwen-max", "label": "Qwen Max" },
    #             { "name": "qwen-plus", "label": "Qwen Plus" }
    #         ]
    #         }
    #     ]
    #     }
@app.post("/api/configs")
def write_configs(request: ConfigRequest):
    try:
        # 读取现有配置（如果存在）
        try:
            with open(CONFIG_FILE, "r") as f:
                current_data = json.load(f)
        except FileNotFoundError:
            current_data = {}

        # 合并新配置
        if request.llm_config is not None:
            current_data["llm_config"] = [config.dict() for config in request.llm_config]
        if request.mcp_config is not None:
            current_data["mcp_config"] = [config.dict() for config in request.mcp_config]

        # 写回文件
        with open(CONFIG_FILE, "w") as f:
            json.dump(current_data, f, indent=2)

        return {"message": "配置已更新"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")