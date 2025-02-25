import os


class RagServiceEnvironment:
    def __init__(self):
        self.IS_API_INSTANCE = os.getenv("DEPLOY_MODE", "web").upper() == "API"
        self.SHOULD_START_WEB = not self.IS_API_INSTANCE


service_environment = RagServiceEnvironment()
