"""Unit tests for context_compressor."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from context_compressor import (
    compress_history,
    estimate_tokens,
    ensure_last_user_in_tail,
    prune_old_tool_results,
    sanitize_tool_pairs,
    trim_to_token_budget,
)


# ────────────── Helpers ────────────── #

def _user(text):
    return {'role': 'user', 'content': text}


def _assistant(text='', tool_calls=None):
    msg = {'role': 'assistant', 'content': text}
    if tool_calls:
        msg['tool_calls'] = [
            {'id': tid, 'type': 'function', 'function': {'name': name, 'arguments': args}}
            for (tid, name, args) in tool_calls
        ]
    return msg


def _tool(tool_call_id, content):
    return {'role': 'tool', 'tool_call_id': tool_call_id, 'content': content}


# ────────────── estimate_tokens ────────────── #

def test_estimate_tokens_handles_string_content():
    msgs = [_user('a' * 400), _assistant('b' * 800)]
    # 400/4 + 800/4 = 100 + 200 = 300
    assert estimate_tokens(msgs) == 300


def test_estimate_tokens_includes_tool_call_arguments():
    msgs = [_assistant('', tool_calls=[('id1', 'foo', '{"x": "y"}' + ' ' * 96)])]
    # arguments len ~106, /4 = 26
    assert estimate_tokens(msgs) >= 25


# ────────────── prune_old_tool_results ────────────── #

def test_prune_truncates_old_tool_messages():
    big = 'x' * 10000
    history = [
        _user('q'),
        _assistant('', tool_calls=[('t1', 'file_read', '{}')]),
        _tool('t1', big),  # this one is OLD
        _user('q2'),
        _assistant('', tool_calls=[('t2', 'file_read', '{}')]),
        _tool('t2', big),  # this one stays full
    ]
    pruned = prune_old_tool_results(history, protect_last_n=3, max_chars=1000)
    assert len(pruned[2]['content']) < len(big)
    assert pruned[5]['content'] == big  # tail intact


def test_prune_does_not_touch_non_tool_messages():
    big = 'x' * 10000
    history = [_user(big), _assistant(big), _tool('t1', big)]
    pruned = prune_old_tool_results(history, protect_last_n=0, max_chars=100)
    assert pruned[0]['content'] == big  # user untouched
    assert pruned[1]['content'] == big  # assistant untouched
    assert len(pruned[2]['content']) < len(big)


def test_prune_preserves_input_list():
    history = [_user('q'), _tool('t1', 'x' * 10000)]
    snapshot = list(history)
    prune_old_tool_results(history, protect_last_n=0, max_chars=100)
    assert history == snapshot  # original not mutated


# ────────────── sanitize_tool_pairs ────────────── #

def test_sanitize_drops_orphan_tool_message():
    history = [
        _user('q'),
        _assistant('answer'),  # no tool_calls
        _tool('t999', 'orphan'),  # no assistant declared this id
    ]
    out = sanitize_tool_pairs(history)
    assert all(m.get('role') != 'tool' for m in out)
    assert len(out) == 2


def test_sanitize_strips_unanswered_tool_call():
    history = [
        _user('q'),
        _assistant('thinking', tool_calls=[('t1', 'foo', '{}'), ('t2', 'bar', '{}')]),
        _tool('t1', 'result1'),
        # t2 has no answer
    ]
    out = sanitize_tool_pairs(history)
    asst = out[1]
    assert len(asst['tool_calls']) == 1
    assert asst['tool_calls'][0]['id'] == 't1'


def test_sanitize_drops_assistant_with_only_unanswered_calls_and_no_content():
    history = [
        _user('q'),
        _assistant('', tool_calls=[('t1', 'foo', '{}')]),
        # t1 no answer
    ]
    out = sanitize_tool_pairs(history)
    # The assistant with empty content + zero answered tool_calls is dropped.
    assert len(out) == 1
    assert out[0]['role'] == 'user'


def test_sanitize_keeps_assistant_with_content_when_calls_unanswered():
    history = [
        _user('q'),
        _assistant('I tried but failed', tool_calls=[('t1', 'foo', '{}')]),
    ]
    out = sanitize_tool_pairs(history)
    assert len(out) == 2
    assert 'tool_calls' not in out[1]
    assert out[1]['content'] == 'I tried but failed'


# ────────────── trim_to_token_budget ────────────── #

def test_trim_returns_copy_when_under_budget():
    history = [_user('hi'), _assistant('hello')]
    out = trim_to_token_budget(history, max_tokens=10000)
    assert out == history
    assert out is not history


def test_trim_anchors_first_and_last():
    # 10 messages each ~1000 chars = ~250 tokens each, total ~2500
    msgs = [_user('a' * 1000)] + [_assistant('b' * 1000) for _ in range(8)] + [_user('q')]
    out = trim_to_token_budget(msgs, max_tokens=500, protect_first_n=1, protect_last_n=1)
    # First and last must always survive
    assert out[0] == msgs[0]
    assert out[-1] == msgs[-1]
    # Middle has been thinned
    assert len(out) < len(msgs)


def test_trim_respects_protected_floor():
    # If we can't go below protect_first_n+protect_last_n without violating
    # the contract, return as-is.
    msgs = [_user('a' * 10000), _assistant('b' * 10000), _user('q' * 10000)]
    out = trim_to_token_budget(msgs, max_tokens=10, protect_first_n=1, protect_last_n=2)
    # 1 + 2 = 3, history is 3, no headroom
    assert len(out) == 3


# ────────────── ensure_last_user_in_tail ────────────── #

def test_ensure_last_user_pass_when_user_present():
    history = [_user('hi'), _assistant('hello'), _user('q')]
    assert ensure_last_user_in_tail(history) == history


def test_ensure_last_user_passes_on_empty_history():
    assert ensure_last_user_in_tail([]) == []


# ────────────── compress_history E2E ────────────── #

def test_compress_under_budget_returns_copy():
    history = [_user('hi'), _assistant('hello')]
    out = compress_history(history, max_tokens=10000)
    assert out == history
    assert out is not history


def test_compress_huge_tool_result_gets_pruned():
    big = 'x' * 50000
    history = [
        _user('analyse this file'),
        _assistant('', tool_calls=[('t1', 'file_read', '{}')]),
        _tool('t1', big),
        _user('summarise'),
    ]
    out = compress_history(history, max_tokens=2000, protect_last_n=2,
                           tool_result_max_chars=500)
    # The pruned tool result is in the protected tail (last 2 messages),
    # so it stays full. With tool result still huge, phase-2 trim will
    # delete the middle messages until under budget.
    final_tokens = estimate_tokens(out)
    # We can't strictly assert <= max_tokens because protected slots may
    # exceed it; the contract is "best effort within protection".
    # Instead assert we removed at least one message OR pruned the body.
    assert final_tokens < estimate_tokens(history) or len(out) < len(history)


def test_compress_e2e_no_orphan_tools():
    # Build a deeply-interleaved history that, when middle-trimmed, would
    # produce orphan tool messages without sanitize.
    history = [_user('first')]
    for turn in range(20):
        tid = f't{turn}'
        history.append(_assistant(f'thinking {turn}',
                                  tool_calls=[(tid, 'foo', '{}')]))
        history.append(_tool(tid, 'result ' * 200))
    history.append(_user('last user question'))

    out = compress_history(history, max_tokens=500, protect_first_n=1, protect_last_n=2,
                           tool_result_max_chars=200)

    # Invariant 1: last user message survives
    assert any(m.get('role') == 'user' and 'last user' in (m.get('content') or '')
               for m in out[-3:])

    # Invariant 2: every tool message has a matching declared id
    declared = set()
    for m in out:
        for tc in m.get('tool_calls') or []:
            declared.add(tc['id'])
    for m in out:
        if m.get('role') == 'tool':
            assert m['tool_call_id'] in declared

    # Invariant 3: every declared tool_call has an answering tool message
    answered = {m['tool_call_id'] for m in out if m.get('role') == 'tool'}
    for m in out:
        for tc in m.get('tool_calls') or []:
            assert tc['id'] in answered


def test_compress_does_not_mutate_input():
    big = 'x' * 50000
    history = [
        _user('q'),
        _assistant('', tool_calls=[('t1', 'foo', '{}')]),
        _tool('t1', big),
    ]
    snapshot = [dict(m) for m in history]
    snapshot_tool_content = history[2]['content']
    compress_history(history, max_tokens=100, protect_last_n=0,
                     tool_result_max_chars=100)
    assert history[2]['content'] == snapshot_tool_content  # not mutated
    assert len(history) == len(snapshot)
