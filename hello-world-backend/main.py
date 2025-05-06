import os
import json
from typing import List
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


class ConfigRequest(BaseModel):
    configs: List[MCPConfig]


# 文件路径
CONFIG_FILE = "config.json"

# 自动创建空配置文件
if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f:
        json.dump([], f)


# 读取配置
@app.get("/api/configs", response_model=List[MCPConfig])
def read_configs():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取配置失败: {str(e)}")


# 写入配置
@app.post("/api/configs")
def write_configs(request: ConfigRequest):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump([config.dict() for config in request.configs], f, indent=2)
        return {"message": "配置已保存"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")
