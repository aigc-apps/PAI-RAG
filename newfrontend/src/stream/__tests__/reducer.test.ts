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

  it("ignores function_call events (tool UI out of scope)", () => {
    const s = fold([
      created("resp_6", "c"),
      { type: "response.function_call_arguments.delta", delta: '{"a":1}' },
      { type: "response.function_call_arguments.done", arguments: '{"a":1}', name: "get" },
      textDelta("done"),
      completed("resp_6", "c"),
    ]);
    expect(s.message.text).toBe("done");
  });

  it("is pure: does not mutate the input state", () => {
    const s0 = initialStreamState("tmp");
    const s1 = reduceStreamEvent(s0, textDelta("a") as any);
    expect(s0.message.text).toBe("");
    expect(s1.message.text).toBe("a");
    expect(s1).not.toBe(s0);
  });
});
