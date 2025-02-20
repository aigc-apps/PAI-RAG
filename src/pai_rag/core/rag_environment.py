import os


class RagServiceEnvironment:
    def __init__(self):
        self.IS_API_INSTANCE = os.getenv("DEPLOY_MODE", "web").upper() == "API"


service_environment = RagServiceEnvironment()
