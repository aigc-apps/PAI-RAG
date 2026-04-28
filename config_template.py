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

# ─────────────────────────── 多用户并发 ───────────────────────────
# 普通登录用户的 file/code 工具会被限制在:
#   WORKSPACE_ROOT/<user_id>/<session_id>/
# 服务级 SERVER_API_KEY 默认仍可使用请求中的 cwd；如需强制服务级调用也隔离，
# 设置 ENFORCE_WORKSPACE_FOR_SERVER = True。
WORKSPACE_ROOT = "./workspaces"
ENFORCE_WORKSPACE_FOR_SERVER = False

# 0 表示不限制。生产环境建议按机器资源和上游模型 QPS 设置。
MAX_GLOBAL_RUNS = 0
MAX_USER_RUNS = 0

# SQLite 日志模式。DELETE 只有主库文件，方便 GUI 查看；WAL 并发更好但会生成 -wal/-shm 文件。
SQLITE_JOURNAL_MODE = "DELETE"

# thread: FastAPI 进程内线程执行，适合本地开发。
# celery: FastAPI 只入队和读 Redis Stream，agent run 由独立 Celery worker 执行。
RUNNER_BACKEND = "thread"
REDIS_URL = "redis://127.0.0.1:6379/0"
CELERY_BROKER_URL = REDIS_URL
RUN_EVENT_TTL_SECONDS = 24 * 60 * 60
ASK_USER_TIMEOUT_SECONDS = 30 * 60
RUN_IDLE_TIMEOUT_SECONDS = 60 * 60

# 多用户生产环境默认不要把共享 memory/global_index.txt 注入普通用户 prompt，
# 避免泄漏全局环境事实。服务级调用不受此开关影响。
# ENABLE_LONG_TERM_MEMORY_FOR_USERS=True 时，普通用户使用独立的
# memory/users/<user_id>/ 长期记忆，不写入全局 memory。
ENABLE_SHARED_MEMORY_FOR_USERS = False
ENABLE_LONG_TERM_MEMORY_FOR_USERS = False

# ─────────────────────────── HTTP 后端 ───────────────────────────
# OpenAI compatible backend exposed by backend/server.py.
BACKEND_BASE_URL = "http://127.0.0.1:8000"

# Optional server-side API key. If non-empty, clients must send:
# Authorization: Bearer <SERVER_API_KEY>
SERVER_API_KEY = ""

# User login token signing secret. Set this to a long random string in any
# shared or persistent deployment. If empty, the server uses a temporary
# development secret and all login tokens become invalid after restart.
AUTH_SECRET = ""
AUTH_TOKEN_TTL_SECONDS = 7 * 24 * 60 * 60

# Browser frontends usually run on a different port from the backend.
# Use ["*"] for local development, or lock this down in production.
SERVER_CORS_ORIGINS = ["*"]

# If True, GET /v1/sessions/{session_id} also returns raw LLM history and
# handler state. Keep False outside local debugging because these fields may
# contain sensitive prompts, file contents, and tool outputs.
EXPOSE_SESSION_DEBUG = False
