"""验证 llm_client._repair_truncated_json:把流式截断的 tool args JSON 补齐。

触发场景:OpenAI-compatible 后端流式工具调用偶发,model 在 finish_reason=tool_calls
状态下少打收尾括号。如果直接吞掉走 _raw_args 兜底,工具拿不到任何字段,典型表现是
ask_user 退化成 "请提供输入：" 占位 —— 整轮被静默吞掉。
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm_client import _repair_truncated_json  # noqa: E402


class RepairTruncatedJsonTests(unittest.TestCase):
    def test_already_valid_passes_through(self):
        self.assertEqual(_repair_truncated_json('{"a": 1}'), {'a': 1})

    def test_missing_closing_brace(self):
        self.assertEqual(_repair_truncated_json('{"a": [1, 2, 3]'), {'a': [1, 2, 3]})

    def test_missing_brace_and_bracket(self):
        self.assertEqual(
            _repair_truncated_json('{"a": [1, 2, 3'),
            {'a': [1, 2, 3]},
        )

    def test_unterminated_string(self):
        self.assertEqual(_repair_truncated_json('{"a": "hel'), {'a': 'hel'})

    def test_dangling_comma(self):
        self.assertEqual(_repair_truncated_json('{"a": 1,'), {'a': 1})

    def test_dangling_colon(self):
        self.assertEqual(_repair_truncated_json('{"a":'), {'a': None})

    def test_garbage_returns_none(self):
        self.assertIsNone(_repair_truncated_json('not json at all'))

    def test_empty_returns_none(self):
        self.assertIsNone(_repair_truncated_json(''))
        self.assertIsNone(_repair_truncated_json('   '))

    def test_brace_inside_string_not_counted(self):
        # 如果误把字符串内的 `}` 当作真收尾,会少补一个外层 `}`。
        self.assertEqual(
            _repair_truncated_json('{"text": "包含 } 字符的字符串"'),
            {'text': '包含 } 字符的字符串'},
        )

    def test_escaped_quote_in_string(self):
        # `\\"` 在字符串里不是字符串结束,后面的 `}` 才是真正的右花括号。
        self.assertEqual(
            _repair_truncated_json(r'{"a": "he said \"hi\""'),
            {'a': 'he said "hi"'},
        )

    def test_mismatched_closer_returns_none(self):
        # `]` 不能 close `{`,这种结构错位放弃修复。
        self.assertIsNone(_repair_truncated_json('{"a": 1]'))

    def test_non_object_array_returns_none(self):
        # 顶层不是 `{` 或 `[` 直接 None,防止把 `12345` 之类东西"修"成数字。
        self.assertIsNone(_repair_truncated_json('"abc'))
        self.assertIsNone(_repair_truncated_json('123'))

    def test_top_level_array_truncation(self):
        self.assertEqual(_repair_truncated_json('[1, 2, 3'), [1, 2, 3])

    def test_real_world_ask_user_truncation(self):
        # 直接搬实测线上 case 的精简版:ask_user 工具 args 漏了外层 `}`。
        s = (
            '{"question": "接口诊断需要参数: cluster-id / instance-id / request-id", '
            '"candidates": ["我有全部参数", "我只有部分参数", "我需要先查 cluster-id 和 instance-id"]'
        )
        out = _repair_truncated_json(s)
        self.assertIsNotNone(out)
        self.assertIn('cluster-id', out['question'])
        self.assertEqual(len(out['candidates']), 3)
        self.assertEqual(out['candidates'][2], '我需要先查 cluster-id 和 instance-id')

    def test_dangling_backslash_then_unterminated_string(self):
        # `"hi\\` —— 末尾单独一个反斜杠,粗暴补 `"` 会变成 `\"` 再次失败。
        # 实现里要先砍掉那个反斜杠再补 `"`。
        self.assertEqual(_repair_truncated_json('{"a": "hi\\'), {'a': 'hi'})


if __name__ == '__main__':
    unittest.main()
