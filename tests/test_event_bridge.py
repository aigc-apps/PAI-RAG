"""SDK ``StreamEvent`` → Responses-wire chunk translation.

Covers three regression-prone behaviours of
``backend.agents_sdk.event_bridge``:

1. **Function-call argument streaming** — when the upstream provider
   emits ``response.output_item.added(function_call)`` plus a series of
   ``response.function_call_arguments.delta`` and a closing
   ``response.function_call_arguments.done``, the bridge forwards them
   as-is so frontends can render arguments token-by-token.
2. **Item-level / raw dedup** — the SDK *also* fires a
   ``RunItemStreamEvent`` carrying a ``ToolCallItem`` for the same
   call. The bridge must not double-emit ``output_item.added`` /
   ``output_item.done``; ``ResponsesStreamState.emitted_item_ids``
   is the gate.
3. **Synthetic reasoning step boundaries** — SDK has no native
   reasoning event; the bridge brackets each "thinking" segment with
   ``response.reasoning_step.started`` / ``.completed`` carrying an
   ``rs_synth_*`` step id. Boundaries: opens on first text delta of a
   segment, closes when a tool call appears or when the assistant
   message finishes.

The fakes deliberately mimic only the duck-typed surface the bridge
reads — full SDK / OpenAI types would drag in pydantic model wiring
that has nothing to do with the contract under test.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from backend.agents_sdk.event_bridge import (
    ResponsesStreamState,
    to_responses_chunk,
)


# ─── fakes ─────────────────────────────────────────────────────────────


def _raw_event(data: Any) -> SimpleNamespace:
    return SimpleNamespace(type='raw_response_event', data=data)


def _item_event(item: Any, name: str = 'tool_called') -> SimpleNamespace:
    return SimpleNamespace(type='run_item_stream_event', item=item, name=name)


def _output_text_delta(delta: str) -> SimpleNamespace:
    return SimpleNamespace(type='response.output_text.delta', delta=delta)


def _output_item_added(item: dict) -> SimpleNamespace:
    return SimpleNamespace(
        type='response.output_item.added', item=item, output_index=0,
    )


def _output_item_done(item: dict) -> SimpleNamespace:
    return SimpleNamespace(
        type='response.output_item.done', item=item, output_index=0,
    )


def _fc_args_delta(item_id: str, delta: str) -> SimpleNamespace:
    return SimpleNamespace(
        type='response.function_call_arguments.delta',
        item_id=item_id, output_index=0, delta=delta,
    )


def _fc_args_done(item_id: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        type='response.function_call_arguments.done',
        item_id=item_id, output_index=0, arguments=arguments,
    )


@dataclass
class _FakeToolCallItem:
    raw_item: dict

    def __post_init__(self) -> None:
        # Match the SDK class name string the bridge keys off.
        self.__class__.__name__ = 'ToolCallItem'

    @property
    def tool_name(self) -> str | None:
        return self.raw_item.get('name')

    @property
    def call_id(self) -> str | None:
        return self.raw_item.get('call_id') or self.raw_item.get('id')


def _make_tool_call_item(*, item_id: str, call_id: str, name: str, arguments: str) -> Any:
    # type-name driven dispatch in the bridge means we need the class name
    # to be exactly 'ToolCallItem'; create it dynamically so isolation is clean.
    cls = type('ToolCallItem', (), {})
    inst = cls()
    inst.raw_item = {
        'id': item_id, 'call_id': call_id, 'name': name,
        'arguments': arguments, 'type': 'function_call',
    }
    inst.tool_name = name
    inst.call_id = call_id
    return inst


def _make_message_output_item() -> Any:
    cls = type('MessageOutputItem', (), {})
    return cls()


# ─── tests ─────────────────────────────────────────────────────────────


class FunctionCallArgumentStreamingTests(unittest.TestCase):
    """Raw streaming path: provider emits added → delta+ → done → output_item.done.
    Bridge forwards every chunk one-for-one and the eventual ``ToolCallItem``
    is suppressed (dedup)."""

    def test_raw_function_call_stream_is_forwarded_verbatim(self):
        state = ResponsesStreamState()
        chunks: list[dict] = []

        item = {
            'id': 'fc_001', 'type': 'function_call',
            'call_id': 'call_001', 'name': 'file_read', 'arguments': '',
            'status': 'in_progress',
        }
        chunks += to_responses_chunk(_raw_event(_output_item_added(item)),
                                     state, response_id='resp_x')
        for piece in ('{"pa', 'th":"', 'README"}'):
            chunks += to_responses_chunk(_raw_event(_fc_args_delta('fc_001', piece)),
                                         state, response_id='resp_x')
        chunks += to_responses_chunk(
            _raw_event(_fc_args_done('fc_001', '{"path":"README"}')),
            state, response_id='resp_x')
        done_item = dict(item)
        done_item['arguments'] = '{"path":"README"}'
        done_item['status'] = 'completed'
        chunks += to_responses_chunk(_raw_event(_output_item_done(done_item)),
                                     state, response_id='resp_x')

        types = [c['type'] for c in chunks]
        self.assertEqual(types, [
            'response.output_item.added',
            'response.function_call_arguments.delta',
            'response.function_call_arguments.delta',
            'response.function_call_arguments.delta',
            'response.function_call_arguments.done',
            'response.output_item.done',
        ])
        # Deltas carry the verbatim partial string.
        deltas = [c['delta'] for c in chunks
                  if c['type'] == 'response.function_call_arguments.delta']
        self.assertEqual(deltas, ['{"pa', 'th":"', 'README"}'])
        # State.output reflects the finalized arguments.
        self.assertEqual(len(state.output), 1)
        self.assertEqual(state.output[0]['arguments'], '{"path":"README"}')
        self.assertEqual(state.output[0]['status'], 'completed')
        # Item id was registered for dedup.
        self.assertIn('fc_001', state.emitted_item_ids)

    def test_subsequent_tool_call_item_is_deduped(self):
        # Drive the raw stream first, then fire the SDK-side ToolCallItem.
        # The bridge must NOT re-emit added/done — the raw stream already did.
        state = ResponsesStreamState()
        item = {
            'id': 'fc_002', 'type': 'function_call',
            'call_id': 'call_002', 'name': 'file_read', 'arguments': '{}',
            'status': 'completed',
        }
        to_responses_chunk(_raw_event(_output_item_added(item)),
                           state, response_id='resp_x')
        to_responses_chunk(_raw_event(_fc_args_done('fc_002', '{}')),
                           state, response_id='resp_x')
        to_responses_chunk(_raw_event(_output_item_done(item)),
                           state, response_id='resp_x')

        sdk_item = _make_tool_call_item(
            item_id='fc_002', call_id='call_002',
            name='file_read', arguments='{}',
        )
        chunks = to_responses_chunk(_item_event(sdk_item),
                                    state, response_id='resp_x')
        self.assertEqual(chunks, [], 'duplicate added/done leaked through dedup')


class ToolCallItemSynthesisTests(unittest.TestCase):
    """Item-level fallback path: provider returned the call without raw
    streaming. Bridge must synthesize ``output_item.added`` + ``.done``."""

    def test_lone_tool_call_item_emits_synthesized_added_done(self):
        state = ResponsesStreamState()
        sdk_item = _make_tool_call_item(
            item_id='fc_003', call_id='call_003',
            name='file_read', arguments='{"path":"x"}',
        )
        chunks = to_responses_chunk(_item_event(sdk_item),
                                    state, response_id='resp_x')
        types = [c['type'] for c in chunks]
        self.assertIn('response.output_item.added', types)
        self.assertIn('response.output_item.done', types)
        added = next(c for c in chunks if c['type'] == 'response.output_item.added')
        self.assertEqual(added['item']['name'], 'file_read')
        self.assertEqual(added['item']['arguments'], '{"path":"x"}')
        # The synthesized id flows from raw_item.id when present.
        self.assertEqual(added['item']['id'], 'fc_003')
        self.assertIn('fc_003', state.emitted_item_ids)


class ReasoningStepBoundaryTests(unittest.TestCase):
    """A 'thinking' segment opens on first text delta and closes when the
    LLM hands off (tool call) or finishes the message."""

    def test_text_then_tool_brackets_one_synthetic_step(self):
        state = ResponsesStreamState()
        chunks: list[dict] = []
        # Plain text is buffered until we know whether it is final answer text
        # or process reasoning before a tool call.
        chunks += to_responses_chunk(_raw_event(_output_text_delta('Let me')),
                                     state, response_id='resp_x')
        chunks += to_responses_chunk(_raw_event(_output_text_delta(' check.')),
                                     state, response_id='resp_x')
        # Tool call closes the step before emitting tool card.
        sdk_item = _make_tool_call_item(
            item_id='fc_010', call_id='call_010',
            name='file_read', arguments='{}',
        )
        chunks += to_responses_chunk(_item_event(sdk_item),
                                     state, response_id='resp_x')

        opens = [c for c in chunks if c['type'] == 'response.reasoning_step.started']
        closes = [c for c in chunks if c['type'] == 'response.reasoning_step.completed']
        self.assertEqual(len(opens), 1, chunks)
        self.assertEqual(len(closes), 1, chunks)
        self.assertEqual(opens[0]['step_id'], closes[0]['step_id'])
        self.assertTrue(opens[0]['step_id'].startswith('rs_synth_'))
        self.assertTrue(opens[0]['synthetic'])
        self.assertEqual(
            ''.join(c['delta'] for c in chunks if c['type'] == 'response.output_text.delta'),
            '',
        )
        self.assertEqual(
            ''.join(c['delta'] for c in chunks if c['type'] == 'response.reasoning_text.delta'),
            'Let me check.',
        )
        self.assertEqual([item['type'] for item in state.output], ['reasoning', 'function_call'])
        self.assertTrue(state.output[0]['metadata']['pairag']['is_process_reasoning'])
        self.assertEqual(state.output[0]['content'][0]['text'], 'Let me check.')

    def test_text_to_message_done_becomes_visible_answer(self):
        state = ResponsesStreamState()
        chunks: list[dict] = []
        chunks += to_responses_chunk(_raw_event(_output_text_delta('hello')),
                                     state, response_id='resp_x')
        msg = _make_message_output_item()
        chunks += to_responses_chunk(_item_event(msg, name='message_output_created'),
                                     state, response_id='resp_x')
        opens = [c for c in chunks if c['type'] == 'response.reasoning_step.started']
        closes = [c for c in chunks if c['type'] == 'response.reasoning_step.completed']
        self.assertEqual(len(opens), 0)
        self.assertEqual(len(closes), 0)
        self.assertEqual(
            [c['type'] for c in chunks],
            [
                'response.output_item.added',
                'response.output_text.delta',
                'response.output_text.done',
                'response.output_item.done',
            ],
        )
        self.assertEqual(chunks[1]['delta'], 'hello')
        self.assertEqual(state.output[0]['content'][0]['text'], 'hello')

    def test_thinking_tags_stream_as_reasoning_not_output_text(self):
        state = ResponsesStreamState()
        chunks: list[dict] = []
        chunks += to_responses_chunk(_raw_event(_output_text_delta('<think')),
                                     state, response_id='resp_x')
        chunks += to_responses_chunk(_raw_event(_output_text_delta('ing>private</thinking>\nAnswer')),
                                     state, response_id='resp_x')
        msg = _make_message_output_item()
        chunks += to_responses_chunk(_item_event(msg), state, response_id='resp_x')

        self.assertEqual(
            ''.join(c['delta'] for c in chunks if c['type'] == 'response.reasoning_text.delta'),
            'private',
        )
        self.assertEqual(
            ''.join(c['delta'] for c in chunks if c['type'] == 'response.output_text.delta'),
            '\nAnswer',
        )
        self.assertNotIn(
            '<thinking>',
            ''.join(c.get('delta', '') for c in chunks),
        )
        self.assertEqual([item['type'] for item in state.output], ['reasoning', 'message'])
        self.assertTrue(state.output[0]['metadata']['pairag']['is_process_reasoning'])
        self.assertEqual(state.output[0]['content'][0]['type'], 'reasoning_text')
        self.assertEqual(state.output[0]['content'][0]['text'], 'private')
        self.assertEqual(state.output[1]['content'][0]['text'], '\nAnswer')

    def test_two_thinking_segments_separated_by_tool_have_distinct_step_ids(self):
        state = ResponsesStreamState()
        all_chunks: list[dict] = []
        # Segment 1: text → tool
        all_chunks += to_responses_chunk(_raw_event(_output_text_delta('think1')),
                                         state, response_id='resp_x')
        sdk_item = _make_tool_call_item(
            item_id='fc_020', call_id='call_020',
            name='file_read', arguments='{}',
        )
        all_chunks += to_responses_chunk(_item_event(sdk_item),
                                         state, response_id='resp_x')
        # Segment 2: text → message done
        all_chunks += to_responses_chunk(_raw_event(_output_text_delta('think2')),
                                         state, response_id='resp_x')
        msg = _make_message_output_item()
        all_chunks += to_responses_chunk(_item_event(msg),
                                         state, response_id='resp_x')

        opens = [c['step_id'] for c in all_chunks
                 if c['type'] == 'response.reasoning_step.started']
        closes = [c['step_id'] for c in all_chunks
                  if c['type'] == 'response.reasoning_step.completed']
        self.assertEqual(len(opens), 1)
        self.assertEqual(len(closes), 1)
        self.assertEqual(opens, closes, 'open/close ids must pair in order')
        self.assertEqual(len(set(opens)), 1)


class PlaceholderItemIdRewriteTests(unittest.TestCase):
    """Some upstreams (qwen-plus) emit `id="__fake_id__"` for every
    function_call. Without rewriting, by-id matching collides — every later
    tool's args overwrite the first tool's, leaving subsequent state.output
    entries with empty arguments. Bridge must rewrite the placeholder to a
    unique id (call_id) so each tool gets its own entry."""

    def test_placeholder_id_is_rewritten_to_call_id(self):
        state = ResponsesStreamState()
        item = {
            'id': '__fake_id__', 'type': 'function_call',
            'call_id': 'call_AAA', 'name': 'use_skill', 'arguments': '',
        }
        chunks = to_responses_chunk(_raw_event(_output_item_added(item)),
                                    state, response_id='resp_x')
        added = next(c for c in chunks if c['type'] == 'response.output_item.added')
        self.assertEqual(added['item']['id'], 'call_AAA')
        self.assertEqual(state.output[0]['id'], 'call_AAA')

    def test_two_calls_with_placeholder_ids_get_distinct_args(self):
        state = ResponsesStreamState()
        # Tool 1
        to_responses_chunk(_raw_event(_output_item_added({
            'id': '__fake_id__', 'type': 'function_call',
            'call_id': 'call_AAA', 'name': 'use_skill', 'arguments': '',
        })), state, response_id='resp_x')
        to_responses_chunk(_raw_event(_fc_args_done('__fake_id__', '{"skill":"a"}')),
                           state, response_id='resp_x')
        to_responses_chunk(_raw_event(_output_item_done({
            'id': '__fake_id__', 'type': 'function_call',
            'call_id': 'call_AAA', 'name': 'use_skill', 'arguments': '{"skill":"a"}',
        })), state, response_id='resp_x')
        # Tool 2 — must NOT overwrite tool 1's args
        to_responses_chunk(_raw_event(_output_item_added({
            'id': '__fake_id__', 'type': 'function_call',
            'call_id': 'call_BBB', 'name': 'code_run', 'arguments': '',
        })), state, response_id='resp_x')
        to_responses_chunk(_raw_event(_fc_args_done('__fake_id__', '{"code":"print(1)"}')),
                           state, response_id='resp_x')
        to_responses_chunk(_raw_event(_output_item_done({
            'id': '__fake_id__', 'type': 'function_call',
            'call_id': 'call_BBB', 'name': 'code_run', 'arguments': '{"code":"print(1)"}',
        })), state, response_id='resp_x')

        fcs = [e for e in state.output if e.get('type') == 'function_call']
        self.assertEqual(len(fcs), 2)
        self.assertEqual(fcs[0]['arguments'], '{"skill":"a"}')
        self.assertEqual(fcs[1]['arguments'], '{"code":"print(1)"}')
        ids = [e['id'] for e in fcs]
        self.assertEqual(len(set(ids)), 2, f'function_call ids must be unique: {ids}')

    def test_args_delta_carries_rewritten_id(self):
        state = ResponsesStreamState()
        to_responses_chunk(_raw_event(_output_item_added({
            'id': '__fake_id__', 'type': 'function_call',
            'call_id': 'call_AAA', 'name': 'use_skill', 'arguments': '',
        })), state, response_id='resp_x')
        chunks = to_responses_chunk(_raw_event(_fc_args_delta('__fake_id__', '{"sk')),
                                    state, response_id='resp_x')
        self.assertEqual(chunks[0]['item_id'], 'call_AAA')


if __name__ == '__main__':
    unittest.main()
