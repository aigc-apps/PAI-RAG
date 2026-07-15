import { describe, it, expect } from "vitest";
import { initialStreamState, reduceStreamEvent } from "../reducer";

// Minimal recorded event sequences mirroring backend/api/protocol/responses_serializer.py.
// Typed loosely as `any` because we only feed the fields the reducer reads.
function fold(events: any[]) {
  let state = initialStreamState("tmp");
  for (const ev of events) state = reduceStreamEvent(state, ev as any);
  return state;
}

const created = (id: string, convId: string | null) => ({
  type: "response.created",
  response: { id, conversation: convId ? { id: convId } : null, status: "in_progress" },
});
const textDelta = (delta: string) => ({ type: "response.output_text.delta", delta });
const reasoningDelta = (delta: string) => ({
  type: "response.reasoning_summary_text.delta",
  delta,
});
const reasoningPartDone = () => ({ type: "response.reasoning_summary_part.done" });
const completed = (id: string, convId: string) => ({
  type: "response.completed",
  response: {
    id,
    conversation: { id: convId },
    status: "completed",
    usage: { input_tokens: 3, output_tokens: 5, total_tokens: 8 },
  },
});
const failed = (id: string, message: string) => ({
  type: "response.failed",
  response: {
    id,
    conversation: null,
    status: "failed",
    error: { code: "server_error", message },
  },
});

describe("reduceStreamEvent", () => {
  it("plain text turn captures ids, text, usage and completed status", () => {
    const s = fold([
      created("resp_1", "conv_1"),
      textDelta("Hell"),
      textDelta("o"),
      completed("resp_1", "conv_1"),
    ]);
    expect(s.responseId).toBe("resp_1");
    expect(s.conversationId).toBe("conv_1");
    expect(s.message.text).toBe("Hello");
    expect(s.message.responseId).toBe("resp_1");
    expect(s.message.status).toBe("completed");
    expect(s.message.usage).toEqual({ input: 3, output: 5, total: 8 });
    expect(s.message.reasoning).toBe("");
    expect(s.message.reasoningStatus).toBe("idle");
  });

  it("reasoning+text turn accumulates reasoning then flips it to done", () => {
    const s = fold([
      created("resp_2", "conv_2"),
      reasoningDelta("think "),
      reasoningDelta("hard"),
      reasoningPartDone(),
      textDelta("answer"),
      completed("resp_2", "conv_2"),
    ]);
    expect(s.message.reasoning).toBe("think hard");
    expect(s.message.reasoningStatus).toBe("done");
    expect(s.message.text).toBe("answer");
    expect(s.message.status).toBe("completed");
    expect(s.message.steps).toEqual([
      { kind: "reasoning", text: "think hard" },
      { kind: "text", text: "answer" },
    ]);
  });

  it("reasoning streaming status is set while deltas arrive", () => {
    const s = fold([created("resp_3", "c"), reasoningDelta("x")]);
    expect(s.message.reasoningStatus).toBe("streaming");
    expect(s.message.status).toBe("streaming");
  });

  it("output_item.done for a reasoning item also marks reasoning done", () => {
    const s = fold([
      created("resp_4", "c"),
      reasoningDelta("x"),
      { type: "response.output_item.done", item: { type: "reasoning" } },
    ]);
    expect(s.message.reasoningStatus).toBe("done");
  });

  it("failed turn sets failed status + error message", () => {
    const s = fold([created("resp_5", "c"), failed("resp_5", "kaboom")]);
    expect(s.message.status).toBe("failed");
    expect(s.message.error).toContain("kaboom");
  });

  it("keeps tool arguments separate from final text", () => {
    const s = fold([
      created("resp_6", "c"),
      {
        type: "response.output_item.added",
        item: { id: "fc_c1", type: "function_call", name: "get", call_id: "c1" },
      },
      { type: "response.function_call_arguments.delta", item_id: "fc_c1", delta: '{"a":1}' },
      { type: "response.function_call_arguments.done", item_id: "fc_c1", arguments: '{"a":1}', name: "get" },
      textDelta("done"),
      completed("resp_6", "c"),
    ]);
    expect(s.message.text).toBe("done");
    expect(s.message.toolCalls).toHaveLength(1);
    expect(s.message.toolCalls[0].arguments).toBe('{"a":1}');
  });

  it("does not duplicate a tool call when output_item.added is replayed", () => {
    const ev = {
      type: "response.output_item.added",
      item: { id: "fc_c1", type: "function_call", name: "get", call_id: "c1" },
    };
    const s = fold([ev, ev]);
    expect(s.message.toolCalls).toHaveLength(1);
    expect(s.message.steps).toEqual([{ kind: "tool", id: "c1" }]);
  });

  it("reasoning_summary_text.done also flips reasoning to done", () => {
    const s = fold([
      created("resp_7", "c"),
      reasoningDelta("partial thought"),
      { type: "response.reasoning_summary_text.done" },
    ]);
    expect(s.message.reasoning).toBe("partial thought");
    expect(s.message.reasoningStatus).toBe("done");
  });

  it("is pure: does not mutate the input state", () => {
    const s0 = initialStreamState("tmp");
    const s1 = reduceStreamEvent(s0, textDelta("a") as any);
    expect(s0.message.text).toBe("");
    expect(s1.message.text).toBe("a");
    expect(s1).not.toBe(s0);
    expect(s1.message).not.toBe(s0.message);
  });
});

describe("reduceStreamEvent — tools", () => {
  it("captures a tool call and its result", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, { type: "response.created", response: { id: "r1", conversation: { id: "c1" } } } as any);
    s = reduceStreamEvent(s, { type: "response.output_item.added", item: { id: "fc_c1", type: "function_call", name: "web_fetch", call_id: "c1" } } as any);
    s = reduceStreamEvent(s, { type: "response.function_call_arguments.delta", item_id: "fc_c1", delta: '{"url":' } as any);
    s = reduceStreamEvent(s, { type: "response.function_call_arguments.done", item_id: "fc_c1", name: "web_fetch", arguments: '{"url":"x"}' } as any);
    s = reduceStreamEvent(s, { type: "response.tool_result", call_id: "c1", output: "PAGE", ok: true } as any);
    s = reduceStreamEvent(s, { type: "response.output_text.delta", delta: "done" } as any);
    expect(s.message.toolCalls).toHaveLength(1);
    const t = s.message.toolCalls[0];
    expect(t.name).toBe("web_fetch");
    expect(t.arguments).toBe('{"url":"x"}');
    expect(t.output).toBe("PAGE");
    expect(t.status).toBe("done");
    expect(s.message.text).toBe("done");
  });

  it("marks a failed tool result as error", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, { type: "response.output_item.added", item: { id: "fc_c1", type: "function_call", name: "web_fetch", call_id: "c1" } } as any);
    s = reduceStreamEvent(s, { type: "response.tool_result", call_id: "c1", output: "boom", ok: false } as any);
    expect(s.message.toolCalls[0].status).toBe("error");
    expect(s.message.toolCalls[0].error).toBe("boom");
  });

  it("folds file artifacts from a tool result onto the tool", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, { type: "response.output_item.added", item: { id: "fc_c1", type: "function_call", name: "publish_artifact", call_id: "c1" } } as any);
    s = reduceStreamEvent(s, {
      type: "response.tool_result",
      call_id: "c1",
      output: "Published report.md",
      ok: true,
      files: [{ id: "tok1", name: "report.md", mime: "text/markdown", size: 12, kind: "markdown" }],
    } as any);
    const t = s.message.toolCalls[0];
    expect(t.files).toHaveLength(1);
    expect(t.files?.[0]).toMatchObject({ id: "tok1", name: "report.md", kind: "markdown" });
  });

  it("folds a stream-only notice from a tool result onto the tool", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, { type: "response.output_item.added", item: { id: "fc_c1", type: "function_call", name: "shell", call_id: "c1" } } as any);
    s = reduceStreamEvent(s, {
      type: "response.tool_result",
      call_id: "c1",
      output: '{"exit_code":1}',
      ok: true,
      notice: { kind: "aliyun_authorization", bound: false, error_code: "InvalidSecurityToken.Expired", interrupt: true },
    } as any);
    const t = s.message.toolCalls[0];
    expect(t.notice).toMatchObject({ kind: "aliyun_authorization", bound: false, interrupt: true });
  });

  it("records a tool's elapsed time from added to result", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, { type: "response.output_item.added", item: { id: "fc_c1", type: "function_call", name: "web_fetch", call_id: "c1" } } as any);
    expect(s.message.toolCalls[0].startedAt).toBeTypeOf("number");
    s = reduceStreamEvent(s, { type: "response.tool_result", call_id: "c1", output: "PAGE", ok: true } as any);
    const d = s.message.toolCalls[0].durationMs;
    expect(d).toBeTypeOf("number");
    expect(d as number).toBeGreaterThanOrEqual(0);
  });

  it("leaves files undefined when the result carries none", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, { type: "response.output_item.added", item: { id: "fc_c1", type: "function_call", name: "web_fetch", call_id: "c1" } } as any);
    s = reduceStreamEvent(s, { type: "response.tool_result", call_id: "c1", output: "PAGE", ok: true, files: [] } as any);
    expect(s.message.toolCalls[0].files).toBeUndefined();
  });

  it("initial message has an empty toolCalls array", () => {
    expect(initialStreamState("x").message.toolCalls).toEqual([]);
  });
});

describe("reduceStreamEvent — narration vs final answer", () => {
  const toolAdded = (callId: string, name: string) => ({
    type: "response.output_item.added",
    item: { id: `fc_${callId}`, type: "function_call", name, call_id: callId },
  });

  it("narration before a tool call is not part of the final answer text", () => {
    const s = fold([
      created("resp_n1", "c"),
      textDelta("我先加载技能。"),
      toolAdded("c1", "load_skill"),
      { type: "response.tool_result", call_id: "c1", output: "OK", ok: true },
      textDelta("这是最终答案。"),
      completed("resp_n1", "c"),
    ]);
    // `text` holds only the trailing run — the answer — not the narration.
    expect(s.message.text).toBe("这是最终答案。");
    // The timeline keeps everything in order: narration, tool, answer.
    expect(s.message.steps).toEqual([
      { kind: "text", text: "我先加载技能。" },
      { kind: "tool", id: "c1" },
      { kind: "text", text: "这是最终答案。" },
    ]);
  });

  it("while a tool is still running there is no final answer yet", () => {
    const s = fold([
      created("resp_n2", "c"),
      textDelta("正在处理…"),
      toolAdded("c1", "load_skill"),
    ]);
    expect(s.message.text).toBe("");
    expect(s.message.steps).toEqual([
      { kind: "text", text: "正在处理…" },
      { kind: "tool", id: "c1" },
    ]);
  });

  it("a plain answer with no tools is a single text step", () => {
    const s = fold([created("resp_n3", "c"), textDelta("hi "), textDelta("there")]);
    expect(s.message.text).toBe("hi there");
    expect(s.message.steps).toEqual([{ kind: "text", text: "hi there" }]);
  });

  it("interleaves reasoning, tools, later reasoning, and the final answer", () => {
    const s = fold([
      created("resp_n4", "c"),
      reasoningDelta("先"),
      reasoningDelta("分析"),
      toolAdded("c1", "search"),
      { type: "response.tool_result", call_id: "c1", output: "OK", ok: true },
      reasoningDelta("检查结果"),
      textDelta("最终答案"),
    ]);

    expect(s.message.reasoning).toBe("先分析检查结果");
    expect(s.message.steps).toEqual([
      { kind: "reasoning", text: "先分析" },
      { kind: "tool", id: "c1" },
      { kind: "reasoning", text: "检查结果" },
      { kind: "text", text: "最终答案" },
    ]);
  });

  it("seeds aggregate legacy content before appending resumed reasoning", () => {
    const initial = initialStreamState("tmp");
    initial.message.reasoning = "旧思考";
    initial.message.text = "旧内容";
    initial.message.steps = undefined;

    const s = reduceStreamEvent(initial, reasoningDelta("新思考") as any);

    expect(s.message.steps).toEqual([
      { kind: "reasoning", text: "旧思考" },
      { kind: "text", text: "旧内容" },
      { kind: "reasoning", text: "新思考" },
    ]);
  });
});

describe("reduceStreamEvent — resilience", () => {
  const withSeq = (ev: any, n: number) => ({ ...ev, sequence_number: n });

  it("captures the max sequence_number across events", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, withSeq({ type: "response.created", response: { id: "r", conversation: { id: "c" } } }, 1) as any);
    s = reduceStreamEvent(s, withSeq(textDelta("hi"), 5) as any);
    s = reduceStreamEvent(s, withSeq({ type: "response.in_progress" }, 3) as any); // older/ignored
    expect(s.lastSequenceNumber).toBe(5);
    expect(s.message.lastSequenceNumber).toBe(5);
  });

  it("maps response.incomplete (status cancelled) to a cancelled message", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, { type: "response.created", response: { id: "r1", conversation: { id: "c1" } } } as any);
    s = reduceStreamEvent(s, textDelta("partial") as any);
    s = reduceStreamEvent(s, {
      type: "response.incomplete",
      response: { id: "r1", conversation: { id: "c1" }, status: "cancelled" },
    } as any);
    expect(s.message.status).toBe("cancelled");
    expect(s.message.text).toBe("partial");
    expect(s.responseId).toBe("r1");
    expect(s.conversationId).toBe("c1");
  });

  it("initial state starts at sequence 0", () => {
    expect(initialStreamState("x").lastSequenceNumber).toBe(0);
  });
});
