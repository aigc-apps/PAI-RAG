import os

# Default region ID
REGION_ID_FROM_ENV = os.environ.get("ALIBABA_CLOUD_REGION_ID")
REGION_ID_FROM_ENV = os.environ.get("REGION_ID", REGION_ID_FROM_ENV)
REGION_ID_FROM_ENV = os.environ.get("REGION", REGION_ID_FROM_ENV) or "cn-hangzhou"


# Default workspace ID
WORKSPACE_ID_FROM_ENV = os.environ.get("PAI_AI_WORKSPACE_ID")
WORKSPACE_ID_FROM_ENV = os.environ.get("PAI_WORKSPACE_ID", WORKSPACE_ID_FROM_ENV)
