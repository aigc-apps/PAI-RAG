import os

ENABLE_TENANT_ID = os.environ.get("ENABLE_TENANT_ID", "false").lower() in ("true", "1", "yes", "y", "on")
DEFAULT_TENANT_ID = "__default_tenant_id__"
