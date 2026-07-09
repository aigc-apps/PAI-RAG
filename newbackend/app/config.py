from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    default_model: str = "openai/gpt-4o-mini"
    db_url: str = "sqlite+aiosqlite:///./data/agent.db"
    store_backend: str = "sql"   # "sql" | "memory"
    agent_name: str = "MiniAgent"
    agent_role: str = "a general-purpose AI assistant"
    search_provider: str = "none"
    search_api_key: str = ""
    search_endpoint: str = ""
    # Knowledge-base retrieval engine. "auto" uses Elasticsearch when
    # `elasticsearch_url` is set (falling back to the local SQL scan if ES is
    # unreachable), "elasticsearch" forces ES (errors when unavailable), "local"
    # forces the built-in scan. ES gives real BM25 full-text + dense_vector kNN
    # hybrid retrieval; the local engine is the zero-dependency dev/test default.
    search_engine: str = "auto"   # "auto" | "elasticsearch" | "local"
    elasticsearch_url: str = ""
    elasticsearch_api_key: str = ""
    elasticsearch_username: str = ""
    elasticsearch_password: str = ""
    elasticsearch_index_prefix: str = "kb"
    elasticsearch_verify_certs: bool = True
    elasticsearch_timeout: int = 30
    skills_dir: str = ""
    config_path: str = "./data/config.yaml"
    models_path: str = "./data/config.yaml"
    app_env: str = "development"
    # Externally-reachable base URL of this deployment (e.g.
    # "https://pai.example.com"), env PUBLIC_BASE_URL. Used to build links we hand
    # out — the ROS one-click template URL and invite links — so they point at the
    # canonical public host instead of whatever Host/proxy header the current
    # request arrived on (often the frontend origin). Empty => fall back to the
    # request's own base_url. Assumes one public origin serves both the SPA and the
    # API; set it to that origin.
    public_base_url: str = ""
    memory_inject_limit: int = 30
    memory_enabled: bool = False
    memory_model: str = ""
    summary_enabled: bool = False
    summary_keep_recent: int = 20
    summary_batch: int = 20
    project_context: str = ""
    # Sandbox file artifacts. `files_url_secret` signs the /v1/files tokens
    # (feature is off / fails closed when empty). `files_nas_local_root` is the
    # backend host's mount of the same NAS export the sandbox sees at /mnt/user;
    # empty => serve by reading bytes back from the live sandbox instead.
    files_url_secret: str = ""
    files_nas_local_root: str = ""
    files_max_bytes: int = 25 * 1024 * 1024
    # Aliyun PAI cross-account authorization. `aliyun_authz_secret` is the HMAC
    # key that derives each user's stable ExternalId (feature fails closed when
    # empty). `aliyun_developer_account_id` (env ALIYUN_DEVELOPER_ACCOUNT_ID) is
    # the trust anchor the customer's RAM role allows — it is NOT hardcoded; it is
    # injected into the ROS template's trust policy at serve time. The developer
    # long-term AK/SK reuse the sandbox provider's AGENTRUN_ACCESS_KEY_ID/_SECRET
    # env. `aliyun_ros_template_url`, when set, overrides the self-hosted template
    # with a public-read OSS URL of authorize-role.yaml for the one-click link.
    aliyun_authz_secret: str = ""
    aliyun_developer_account_id: str = ""
    aliyun_default_region: str = "cn-hangzhou"
    aliyun_ros_template_url: str = ""
    # Authentication. `jwt_secret` signs the HS256 access tokens AND the invite
    # tokens; auth fails closed (503) when it's empty, so every deployment must
    # set it. `jwt_ttl_minutes` bounds an access token's lifetime — kept modest
    # because status (active/disabled) is re-checked from the DB on every request,
    # so a disabled user's still-valid token stops working within one TTL at
    # worst. `cookie_secure` must be true behind HTTPS in production (the browser
    # session cookie). `admin_bootstrap_token`, when set, is additionally required
    # by the first-run admin bootstrap as a second factor.
    jwt_secret: str = ""
    jwt_ttl_minutes: int = 720
    invite_ttl_hours: int = 72
    cookie_secure: bool = False
    admin_bootstrap_token: str = ""


def get_settings() -> Settings:
    return Settings()
