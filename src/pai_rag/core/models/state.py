import os


class FileServiceState:
    def __init__(self, key):
        self.state_key = key
        self.state_value = -1
        self.state_value = self.check_state()

    def check_state(self):
        if not os.path.exists(self.state_key):
            return 0
        mtime = os.path.getmtime(self.state_key)
        if mtime != self.state_value:
            return mtime
        return 0

    def update_state(self, new_value):
        self.state_value = new_value
