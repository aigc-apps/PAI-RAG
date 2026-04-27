# 复制为 config.py 后填入真实凭证
# cp config_template.py config.py

# ─────────────────────────── 鉴权 ───────────────────────────
# 通过标准 OpenAI 兼容协议鉴权（Authorization: Bearer <API_KEY>）
API_KEY = "sk-<your-api-key>"

# ─────────────────────────── 端点 & 模型 ───────────────────────────
# 默认：阿里云 DashScope（Qwen）— OpenAI 兼容模式
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"   # 可选: qwen-max / qwen-plus / qwen-turbo / qwen3-coder-plus 等

# 其它兼容端点（按需取消注释切换）：
#
# A. OpenAI 官方：
#    API_BASE = "https://api.openai.com/v1"
#    MODEL    = "gpt-4o"   # 或 gpt-4o-mini / o1 等
#
# B. DeepSeek：
#    API_BASE = "https://api.deepseek.com/v1"
#    MODEL    = "deepseek-chat"   # 或 deepseek-reasoner
#
# C. 本地 vLLM / Ollama / LM Studio（OpenAI 兼容服务器）：
#    API_BASE = "http://127.0.0.1:8000/v1"
#    MODEL    = "<your-served-model>"
#    API_KEY  = "EMPTY"   # 多数本地服务器无需鉴权，填占位串即可
#
# D. OpenRouter（聚合多家）：
#    API_BASE = "https://openrouter.ai/api/v1"
#    MODEL    = "anthropic/claude-3.5-sonnet"   # 等

# ─────────────────────────── 容量 ───────────────────────────
MAX_TOKENS = 8192                  # 单次回复最大 token
MAX_TURNS = 40                     # 单任务最大轮数
HISTORY_TRIM_TOKENS = 80000        # 历史超过此估算值后裁剪最早消息
TIMEOUT = 300                      # 秒（请求总超时）

# ─────────────────────────── HTTP 后端 ───────────────────────────
# OpenAI compatible backend exposed by backend/server.py.
BACKEND_BASE_URL = "http://127.0.0.1:8000"

# Optional server-side API key. If non-empty, clients must send:
# Authorization: Bearer <SERVER_API_KEY>
SERVER_API_KEY = ""

# Browser frontends usually run on a different port from the backend.
# Use ["*"] for local development, or lock this down in production.
SERVER_CORS_ORIGINS = ["*"]

# If True, GET /v1/sessions/{session_id} also returns raw LLM history and
# handler state. Keep False outside local debugging because these fields may
# contain sensitive prompts, file contents, and tool outputs.
EXPOSE_SESSION_DEBUG = False
