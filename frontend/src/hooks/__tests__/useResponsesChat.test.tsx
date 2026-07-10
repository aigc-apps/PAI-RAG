import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

vi.mock("../../api/client", () => ({ streamResponse: vi.fn() }));
vi.mock("../../api/responses", () => ({ cancelResponse: vi.fn(), streamResume: vi.fn() }));
vi.mock("../../store/conversations", () => ({
  useConversationsStore: { getState: () => ({ refresh: vi.fn(), select: vi.fn() }) },
}));

import { useResponsesChat } from "../useResponsesChat";
import { activeRuntime, useChatStore } from "../../store/chat";
import type { ChatMessage } from "../../types";
import * as client from "../../api/client";
import * as responsesApi from "../../api/responses";

function streamOf(events: any[]): AsyncIterable<any> {
  return {
    async *[Symbol.asyncIterator]() {
      for (const e of events) yield e;
    },
  };
}

const active = () => activeRuntime(useChatStore.getState())!;
function seedActive(msgs: ChatMessage[]) {
  const key = useChatStore.getState().activeKey;
  for (const m of msgs) useChatStore.getState().appendMessage(key, m);
}

beforeEach(() => {
  vi.clearAllMocks();
  useChatStore.getState().reset();
});

describe("useResponsesChat (resilient)", () => {
  it("send runs in background and advances anchors on completion", async () => {
    (client.streamResponse as any).mockReturnValue(
      streamOf([
        { type: "response.created", response: { id: "resp_1", conversation: { id: "conv_1" } }, sequence_number: 1 },
        { type: "response.output_text.delta", delta: "hi", sequence_number: 2 },
        { type: "response.completed", response: { id: "resp_1", conversation: { id: "conv_1" }, status: "completed" }, sequence_number: 3 },
      ])
    );
    const { result } = renderHook(() => useResponsesChat());
    await act(async () => { await result.current.send("hello"); });
    expect(active().messages[1].text).toBe("hi");
    expect(active().messages[1].status).toBe("completed");
    expect(active().conversationId).toBe("conv_1");
    expect(active().lastResponseId).toBe("resp_1");
    const params = (client.streamResponse as any).mock.calls[0][0];
    expect(params.background).toBe(true);
    // Identity is server-derived from the session; no user_id is sent.
    expect(params.user_id).toBeUndefined();
  });

  it("stop cancels server-side once the response id is known", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    (client.streamResponse as any).mockReturnValue({
      async *[Symbol.asyncIterator]() {
        yield { type: "response.created", response: { id: "resp_9", conversation: { id: "c" } }, sequence_number: 1 };
        yield { type: "response.output_text.delta", delta: "partial", sequence_number: 2 };
        await gate;
        yield { type: "response.incomplete", response: { id: "resp_9", conversation: { id: "c" }, status: "cancelled" }, sequence_number: 3 };
      },
    });
    const { result } = renderHook(() => useResponsesChat());
    let p: Promise<void>;
    await act(async () => { p = result.current.send("hello"); await Promise.resolve(); });
    act(() => result.current.stop());
    expect(responsesApi.cancelResponse).toHaveBeenCalledWith("resp_9");
    release();
    await act(async () => { await p; });
    expect(active().messages[1].status).toBe("cancelled");
    expect(active().messages[1].text).toBe("partial");
    // cancelled turn is continuable: anchors advanced
    expect(active().conversationId).toBe("c");
    expect(active().lastResponseId).toBe("resp_9");
  });

  it("stop before the response id is known defers the cancel until response.created", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    (client.streamResponse as any).mockReturnValue({
      async *[Symbol.asyncIterator]() {
        await gate; // hold BEFORE the first event so stop() runs with no id yet
        yield { type: "response.created", response: { id: "resp_d", conversation: { id: "c" } }, sequence_number: 1 };
        yield { type: "response.completed", response: { id: "resp_d", conversation: { id: "c" }, status: "completed" }, sequence_number: 2 };
      },
    });
    const { result } = renderHook(() => useResponsesChat());
    let p: Promise<void>;
    await act(async () => {
      p = result.current.send("hello");
      await Promise.resolve();
    });
    // id not known yet -> stop must defer, not call cancel
    act(() => result.current.stop());
    expect(responsesApi.cancelResponse).not.toHaveBeenCalled();
    release();
    await act(async () => {
      await p;
    });
    // once response.created yielded the id, the deferred cancel fires exactly once
    expect(responsesApi.cancelResponse).toHaveBeenCalledTimes(1);
    expect(responsesApi.cancelResponse).toHaveBeenCalledWith("resp_d");
  });

  it("resumeIfInterrupted resumes a streaming message from its cursor", async () => {
    // seed a half-streamed assistant message in the active runtime
    seedActive([
      { id: "u", role: "user", text: "q", reasoning: "", reasoningStatus: "idle", status: "completed", toolCalls: [] },
      { id: "resp_5", role: "assistant", text: "par", reasoning: "", reasoningStatus: "idle", status: "streaming", responseId: "resp_5", lastSequenceNumber: 4, toolCalls: [] },
    ]);
    (responsesApi.streamResume as any).mockReturnValue(
      streamOf([
        { type: "response.output_text.delta", delta: "tial", sequence_number: 5 },
        { type: "response.completed", response: { id: "resp_5", conversation: { id: "c5" }, status: "completed" }, sequence_number: 6 },
      ])
    );
    const { result } = renderHook(() => useResponsesChat());
    await act(async () => { await result.current.resumeIfInterrupted(); });
    expect(responsesApi.streamResume).toHaveBeenCalledWith("resp_5", 4, expect.anything());
    expect(active().messages[1].text).toBe("partial");
    expect(active().messages[1].status).toBe("completed");
    expect(active().lastResponseId).toBe("resp_5");
  });

  it("resumeIfInterrupted is a no-op when the last message is not streaming", async () => {
    seedActive([{ id: "a", role: "assistant", text: "done", reasoning: "", reasoningStatus: "idle", status: "completed", responseId: "r", toolCalls: [] }]);
    const { result } = renderHook(() => useResponsesChat());
    await act(async () => { await result.current.resumeIfInterrupted(); });
    expect(responsesApi.streamResume).not.toHaveBeenCalled();
  });

  it("resumeIfInterrupted is a no-op while a send is in flight", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    (client.streamResponse as any).mockReturnValue({
      async *[Symbol.asyncIterator]() {
        yield { type: "response.created", response: { id: "resp_x", conversation: { id: "c" } }, sequence_number: 1 };
        yield { type: "response.output_text.delta", delta: "hi", sequence_number: 2 };
        await gate;
        yield { type: "response.completed", response: { id: "resp_x", conversation: { id: "c" }, status: "completed" }, sequence_number: 3 };
      },
    });
    const { result } = renderHook(() => useResponsesChat());
    let p: Promise<void>;
    await act(async () => { p = result.current.send("hello"); await Promise.resolve(); });
    await act(async () => { await result.current.resumeIfInterrupted(); });
    expect(responsesApi.streamResume).not.toHaveBeenCalled();
    release();
    await act(async () => { await p; });
  });

  it("a mid-stream transport error keeps the bubble resumable and resume recovers it", async () => {
    (client.streamResponse as any).mockReturnValue({
      async *[Symbol.asyncIterator]() {
        yield { type: "response.created", response: { id: "resp_r", conversation: { id: "c" } }, sequence_number: 1 };
        yield { type: "response.output_text.delta", delta: "par", sequence_number: 2 };
        throw new Error("network dropped");
      },
    });
    const { result } = renderHook(() => useResponsesChat());
    await act(async () => { await result.current.send("hello"); });
    expect(active().messages[1].status).toBe("streaming"); // not "failed" — run started, resumable
    expect(active().messages[1].text).toBe("par");
    (responsesApi.streamResume as any).mockReturnValue(streamOf([
      { type: "response.output_text.delta", delta: "tial", sequence_number: 3 },
      { type: "response.completed", response: { id: "resp_r", conversation: { id: "c" }, status: "completed" }, sequence_number: 4 },
    ]));
    await act(async () => { await result.current.resumeIfInterrupted(); });
    expect(active().messages[1].text).toBe("partial");
    expect(active().messages[1].status).toBe("completed");
    expect(responsesApi.streamResume).toHaveBeenCalledWith("resp_r", 2, expect.anything());
  });

  it("switching to another conversation aborts the local loop without cross-writing", async () => {
    // A: a long-running stream we'll leave mid-flight.
    let releaseA: () => void = () => {};
    const gateA = new Promise<void>((r) => (releaseA = r));
    (client.streamResponse as any).mockReturnValue({
      async *[Symbol.asyncIterator]() {
        yield { type: "response.created", response: { id: "resp_A", conversation: { id: "cA" } }, sequence_number: 1 };
        yield { type: "response.output_text.delta", delta: "A-partial", sequence_number: 2 };
        await gateA; // never released before we switch away
        yield { type: "response.completed", response: { id: "resp_A", conversation: { id: "cA" }, status: "completed" }, sequence_number: 3 };
      },
    });
    const { result } = renderHook(() => useResponsesChat());
    const keyA = useChatStore.getState().activeKey;
    let pA: Promise<void>;
    await act(async () => { pA = result.current.send("hi from A"); await Promise.resolve(); await Promise.resolve(); });

    // Switch to a brand-new conversation B. The effect aborts A's local loop.
    await act(async () => { useChatStore.getState().newDraft(); });
    const keyB = useChatStore.getState().activeKey;
    expect(keyB).not.toBe(keyA);

    // A's partial text is preserved in its own slice; B is untouched/empty.
    expect(useChatStore.getState().runtimes[keyA].messages[1].text).toBe("A-partial");
    expect(useChatStore.getState().runtimes[keyA].messages[1].status).toBe("streaming"); // resumable
    expect(active().messages).toHaveLength(0);

    releaseA();
    await act(async () => { await pA; });
    // Even after A's generator resumes post-abort, B (the visible one) is clean.
    expect(active().messages).toHaveLength(0);
  });
});
