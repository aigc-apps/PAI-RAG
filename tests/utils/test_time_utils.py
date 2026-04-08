import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from utils.time_utils import get_current_time_str, get_current_time_str_zh


class TestTimeUtils:
    def test_get_current_time_str_format(self):
        result = get_current_time_str()
        # Format: "2024-01-15 10:30:00 Monday"
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+", result)

    def test_get_current_time_str_zh_format(self):
        result = get_current_time_str_zh()
        # Format: "2024年01月15日 10:30:00 Monday"
        assert re.match(r"\d{4}年\d{2}月\d{2}日 \d{2}:\d{2}:\d{2} \w+", result)
