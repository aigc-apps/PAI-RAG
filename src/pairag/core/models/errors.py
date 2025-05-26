class UserInputError(Exception):
    def __init__(self, msg: str):
        self.msg = msg


class ServiceError(Exception):
    def __init__(self, msg: str):
        self.msg = msg
