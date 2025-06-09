import os
import json
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from core.chat import handle_chat

from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, set_key, find_dotenv
from trace.tracing_config import TracingConfig
from trace.base import init_instrument

load_dotenv()
init_instrument()

app = FastAPI()

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有的HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)


# 数据模型
class MCPConfig(BaseModel):
    id: str
    name: str
    url: str
    type: str
    active: bool


class LLMConfig(BaseModel):
    id: str
    model_id: str = None
    model_name: str = None
    base_url: str = None
    api_key: str = None
    is_active: bool = True


class LLMConfigRequest(BaseModel):
    llm_config: LLMConfig = None


class MCPConfigRequest(BaseModel):
    mcp_config: MCPConfig = None


class SearchConfigRequest(BaseModel):
    aliyun_ak: str = None
    aliyun_sk: str = None


class ConfigRequest(BaseModel):
    llm_config: Optional[List[LLMConfig]] = None
    mcp_config: Optional[List[MCPConfig]] = None


class WebSearchRequest(BaseModel):
    query: str
    count: int = 10
    lang: str = "zh-CN"
    time_range: str = "OneMonth"  # OneMonth, OneWeek, OneDay, OneYear, NoLimit


CONFIG_FILE = "config.json"

# 自动创建默认配置文件
if not os.path.exists(CONFIG_FILE):
    default_config = {"llm_config": [], "mcp_config": []}
    with open(CONFIG_FILE, "w") as f:
        json.dump(default_config, f, indent=2)


if not os.path.exists(".env"):
    with open(".env", "w") as f:
        f.write("")


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
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取配置失败: {str(e)}")

    url_and_source_map = {
        "https://api.openai.com/v1": "OpenAI",
        "https://dashscope.aliyuncs.com/compatible-mode/v1": "Qwen",
    }

    grouped = {}

    # 处理 llm_config
    for config in data.get("llm_config", []):
        base_url = config.get("base_url")
        model_id = config.get("model_id")
        is_active = config.get("is_active", False)
        source = None
        if base_url in url_and_source_map:
            source = url_and_source_map[base_url]
        else:
            source = "Others"

        if not source or not model_id or not is_active:
            continue  # 跳过无效配置

        if source not in grouped:
            grouped[source] = []

        grouped[source].append({"name": model_id, "id": config.get("id")})

    return {
        "groups": [
            {"id": source.lower(), "label": source, "models": models}
            for source, models in grouped.items()
        ]
    }


@app.post("/api/add_llm")
async def add_llm(req: LLMConfigRequest):
    try:
        # 读取现有配置（如果存在）
        try:
            with open(CONFIG_FILE, "r") as f:
                current_data = json.load(f)
        except FileNotFoundError:
            current_data = {}

        # 处理 LLM 配置（追加模式）
        if "llm_config" not in current_data:
            current_data["llm_config"] = []

        # 检查是否已经存在相同的配置
        is_existing = False
        for existing_config in current_data["llm_config"]:
            if existing_config["id"] == req.llm_config.id:
                existing_config["model_id"] = req.llm_config.model_id
                existing_config["model_name"] = req.llm_config.model_name
                existing_config["base_url"] = req.llm_config.base_url
                existing_config["api_key"] = req.llm_config.api_key
                existing_config["is_active"] = req.llm_config.is_active
                is_existing = True
                break
        if not is_existing:
            current_data["llm_config"].append(req.llm_config.dict())

        # 写回文件
        with open(CONFIG_FILE, "w") as f:
            json.dump(current_data, f, indent=2)

        return {"message": "配置已更新"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")


@app.delete("/api/delete_llm/{llm_id}")
async def delete_llm(llm_id: str):
    try:
        # 读取现有配置
        try:
            with open(CONFIG_FILE, "r") as f:
                current_data = json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="配置文件未找到")

        # 确保 llm_config 存在
        if "llm_config" not in current_data:
            raise HTTPException(status_code=400, detail="无 LLM 配置可删除")

        # 过滤掉要删除的配置
        llm_configs = current_data["llm_config"]
        updated_configs = [cfg for cfg in llm_configs if cfg.get("id") != llm_id]

        # 如果没有配置被删除，说明未找到对应 ID
        if len(updated_configs) == len(llm_configs):
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {llm_id} 的 LLM 配置"
            )

        # 更新配置
        current_data["llm_config"] = updated_configs

        # 写回文件
        with open(CONFIG_FILE, "w") as f:
            json.dump(current_data, f, indent=2)

        return {"message": f"LLM 配置 (ID: {llm_id}) 已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@app.post("/api/add_mcp")
async def add_mcp(req: MCPConfigRequest):
    try:
        # 读取现有配置（如果存在）
        try:
            with open(CONFIG_FILE, "r") as f:
                current_data = json.load(f)
        except FileNotFoundError:
            current_data = {}

        # 处理 mcp 配置（追加模式）
        if "mcp_config" not in current_data:
            current_data["mcp_config"] = []

        # 检查是否已经存在相同的配置
        is_existing = False
        for existing_config in current_data["mcp_config"]:
            if existing_config["id"] == req.mcp_config.id:
                existing_config["name"] = req.mcp_config.name
                existing_config["url"] = req.mcp_config.url
                existing_config["type"] = req.mcp_config.type
                existing_config["active"] = req.mcp_config.active
                is_existing = True
                break
        if not is_existing:
            current_data["mcp_config"].append(req.mcp_config.dict())

        # 写回文件
        with open(CONFIG_FILE, "w") as f:
            json.dump(current_data, f, indent=2)

        return {"message": "配置已更新"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")


@app.delete("/api/delete_mcp/{mcp_id}")
async def delete_mcp(mcp_id: str):
    try:
        # 读取现有配置
        try:
            with open(CONFIG_FILE, "r") as f:
                current_data = json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="配置文件未找到")

        # 确保 mcp_config 存在
        if "mcp_config" not in current_data:
            raise HTTPException(status_code=400, detail="无 LLM 配置可删除")

        # 过滤掉要删除的配置
        mcp_configs = current_data["mcp_config"]
        updated_configs = [cfg for cfg in mcp_configs if cfg.get("id") != mcp_id]

        # 如果没有配置被删除，说明未找到对应 ID
        if len(updated_configs) == len(mcp_configs):
            raise HTTPException(
                status_code=404, detail=f"未找到 ID 为 {mcp_id} 的 LLM 配置"
            )

        # 更新配置
        current_data["mcp_config"] = updated_configs

        # 写回文件
        with open(CONFIG_FILE, "w") as f:
            json.dump(current_data, f, indent=2)

        return {"message": f"LLM 配置 (ID: {mcp_id}) 已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@app.get("/api/search_config")
def get_search_config():
    try:
        # 加载现有 .env 文件（可选）
        load_dotenv()

        return {
            "ACCESS_KEY_ID": os.getenv("ACCESS_KEY_ID"),
            "ACCESS_KEY_SECRET": os.getenv("ACCESS_KEY_SECRET"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取失败: {str(e)}")


@app.post("/api/search_config")
def update_search_config(request: SearchConfigRequest):
    try:
        # 加载现有 .env 文件（可选）
        load_dotenv()

        # 写入新的 AK/SK 到 .env 文件
        set_key(find_dotenv(), "ACCESS_KEY_ID", request.aliyun_ak)
        set_key(find_dotenv(), "ACCESS_KEY_SECRET", request.aliyun_sk)
        os.environ["ACCESS_KEY_ID"] = request.aliyun_ak
        os.environ["ACCESS_KEY_SECRET"] = request.aliyun_sk

        return {"message": "配置已更新"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


@app.get("/api/tracing_config")
def get_tracing_config():
    try:
        # 加载现有 .env 文件（可选）
        load_dotenv()

        return {
            "TRACING_ENDPOINT": os.getenv("TRACING_ENDPOINT"),
            "TRACING_TOKEN": os.getenv("TRACING_TOKEN"),
            "TRACING_SERVICE_NAME": os.getenv("TRACING_SERVICE_NAME"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取失败: {str(e)}")


@app.post("/api/tracing_config")
def update_tracing_config(tracing_config: TracingConfig):
    try:
        # 加载现有 .env 文件（可选）
        load_dotenv()

        # 写入新的 AK/SK 到 .env 文件
        set_key(find_dotenv(), "TRACING_ENDPOINT", tracing_config.endpoint)
        set_key(find_dotenv(), "TRACING_TOKEN", tracing_config.token)
        set_key(find_dotenv(), "TRACING_SERVICE_NAME", tracing_config.service_name)

        os.environ["TRACING_ENDPOINT"] = tracing_config.endpoint
        os.environ["TRACING_TOKEN"] = tracing_config.token
        os.environ["TRACING_SERVICE_NAME"] = tracing_config.service_name

        init_instrument()
        return {"message": "配置已更新"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


@app.post("/api/chat")
async def chat(request: Request):
    return await handle_chat(request)
