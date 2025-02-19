import os


class RagServiceEnvironment:
    def __init__(self):
        self.IS_MULTIPLE_INSTANCE = (
            os.getenv("DEPLOY_MODE", "single").upper() == "MULTIPLE"
        )


service_environment = RagServiceEnvironment()
