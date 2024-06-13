from datetime import datetime
import pytz

# 创建北京时区的变量
beijing_tz = pytz.timezone("Asia/Shanghai")


class MyUploadFile:
    def __init__(self, file_name, task_id):
        self.file_name = file_name
        self.task_id = task_id
        self.start_time = datetime.now(beijing_tz)
        self.end_time = None
        self.duration = None
        self.state = None
        self.finished = False

    def update_process_duration(self):
        if not self.finished:
            self.end_time = datetime.now(beijing_tz)
            self.duration = (self.end_time - self.start_time).total_seconds()
            return self.duration

    def update_state(self, state):
        self.state = state

    def is_finished(self):
        self.finished = True

    def __info__(self):
        return [
            self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            self.duration,
            self.state,
        ]
