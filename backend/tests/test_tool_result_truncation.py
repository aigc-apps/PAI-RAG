# ruff: noqa: E401, E402
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.tool_result_truncation import (
    smart_truncate,
    _shrink,
    _estimate,
    TRUNCATED_MARKER,
    ARRAY_HEAD,
    ARRAY_TAIL,
    OBJECT_MAX_KEYS,
)


# --------------------------------------------------------------------------- #
# _shrink — structural rules (deterministic, no tokenizer involved)
# --------------------------------------------------------------------------- #
def test_shrink_long_array_keeps_head_and_tail():
    out = _shrink({"items": list(range(200)), "total": 200})
    items = out["items"]
    assert len(items) == ARRAY_HEAD + 1 + ARRAY_TAIL
    assert items[:ARRAY_HEAD] == list(range(ARRAY_HEAD))            # head intact
    assert "elements omitted" in items[ARRAY_HEAD]                  # middle sentinel
    assert items[ARRAY_HEAD + 1:] == list(range(200 - ARRAY_TAIL, 200))  # tail intact
    assert out["total"] == 200                                     # scalar sibling kept


def test_shrink_short_array_untouched():
    node = {"items": [1, 2, 3]}
    assert _shrink(node) == node


def test_shrink_long_string_head_tail():
    s = "START" + "x" * 5000 + "END"
    out = _shrink({"blob": s, "keep": "short"})
    assert out["keep"] == "short"                                  # short string kept whole
    assert out["blob"].startswith("START") and out["blob"].endswith("END")
    assert "chars omitted" in out["blob"]
    assert len(out["blob"]) < len(s)


def test_shrink_many_keys_keeps_head_tail_and_sentinel():
    node = {f"k{i}": i for i in range(OBJECT_MAX_KEYS + 40)}
    out = _shrink(node)
    assert "__omitted__" in out and "keys omitted" in out["__omitted__"]
    assert "k0" in out                                             # first key kept
    assert f"k{OBJECT_MAX_KEYS + 39}" in out                       # last key kept
    # a normal-sized object keeps every key
    assert _shrink({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_shrink_nested_json_in_string():
    inner = {"Services": [{"id": i, "cfg": "y" * 300} for i in range(100)],
             "TotalCount": 100}
    node = {"exit_code": 0, "stdout": json.dumps(inner), "stderr": ""}
    out = _shrink(node)
    assert out["exit_code"] == 0
    assert isinstance(out["stdout"], str)                          # shape preserved (still a string)
    reparsed = json.loads(out["stdout"])
    assert reparsed["TotalCount"] == 100                           # tail scalar survived
    assert len(reparsed["Services"]) == ARRAY_HEAD + 1 + ARRAY_TAIL
    assert "y" * 300 not in out["stdout"]                          # long cfg strings truncated


# --------------------------------------------------------------------------- #
# smart_truncate — end to end (character-estimated)
# --------------------------------------------------------------------------- #
def test_small_content_returned_unchanged():
    assert smart_truncate("hello world", 5000) == "hello world"
    assert smart_truncate("", 5000) == ""


def test_plain_text_keeps_head_and_tail():
    content = "HEADSTART " + ("filler " * 20000) + " TAILEND"
    out = smart_truncate(content, 100)
    assert out.endswith(TRUNCATED_MARKER)
    assert "HEADSTART" in out                                      # head preserved
    assert "TAILEND" in out                                        # tail preserved (the point)
    assert "tokens omitted" in out                                 # middle elided
    assert _estimate(out) < _estimate(content)


def test_json_reduces_fits_and_preserves_tail():
    inner = {"Services": [{"id": i, "cfg": "z" * 400} for i in range(100)],
             "TotalCount": 103}
    content = json.dumps({"exit_code": 0, "stdout": json.dumps(inner), "stderr": ""})
    cap = 200
    out = smart_truncate(content, cap)
    assert out.endswith(TRUNCATED_MARKER)
    assert _estimate(out) < _estimate(content)         # large reduction
    assert _estimate(out) <= cap + 40                  # backstop fits (+marker overhead)
    assert "Services" in out                                       # head structure survives
    assert "TotalCount" in out                                     # tail survives backstop


def test_non_json_over_cap_uses_text_path():
    content = "line-of-logs\n" * 5000  # not valid JSON
    out = smart_truncate(content, 50)
    assert out.endswith(TRUNCATED_MARKER)
    assert "tokens omitted" in out
    assert _estimate(out) <= 50 + 40


def test_text_head_tail_is_character_estimated_only():
    import agent.tool_result_truncation as t
    assert not hasattr(t, "estimate_tokens_in_text")
    assert not hasattr(t, "truncate")
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10
    out = t._text_head_tail(text, max_tokens=10)
    assert out.startswith("ABCDEFGHIJKLMNOPQRST")
    assert out.endswith("VWXYZ")
    assert "tokens omitted" in out
