import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

vi.mock("../../api/client", () => ({ streamResponse: vi.fn() }));
vi.mock("../../api/responses", () => ({ cancelResponse: vi.fn(), streamResume: vi.fn() }));
vi.mock("../../lib/user", () => ({ getUserId: () => "u1" }));
vi.mock("../../store/conversations", () => ({
  useConversationsStore: { getState: () => ({ refresh: vi.fn() }) },
}));

import { useResponsesChat } from "../useResponsesChat";
import { useChatStore } from "../../store/chat";
import * as client from "../../api/client";
import * as responsesApi from "../../api/responses";

function streamOf(events: any[]): AsyncIterable<any> {
  return {
    async *[Symbol.asyncIterator]() {
      for (const e of events) yield e;
    },
  };
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
    const st = useChatStore.getState();
    expect(st.messages[1].text).toBe("hi");
    expect(st.messages[1].status).toBe("completed");
    expect(st.conversationId).toBe("conv_1");
    expect(st.lastResponseId).toBe("resp_1");
    const params = (client.streamResponse as any).mock.calls[0][0];
    expect(params.background).toBe(true);
    expect(params.user_id).toBe("u1");
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
    const st = useChatStore.getState();
    expect(st.messages[1].status).toBe("cancelled");
    expect(st.messages[1].text).toBe("partial");
    // cancelled turn is continuable: anchors advanced
    expect(st.conversationId).toBe("c");
    expect(st.lastResponseId).toBe("resp_9");
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
    // seed a half-streamed assistant message in the store
    useChatStore.setState({
      messages: [
        { id: "u", role: "user", text: "q", reasoning: "", reasoningStatus: "idle", status: "completed" },
        { id: "resp_5", role: "assistant", text: "par", reasoning: "", reasoningStatus: "idle", status: "streaming", responseId: "resp_5", lastSequenceNumber: 4 },
      ],
    });
    (responsesApi.streamResume as any).mockReturnValue(
      streamOf([
        { type: "response.output_text.delta", delta: "tial", sequence_number: 5 },
        { type: "response.completed", response: { id: "resp_5", conversation: { id: "c5" }, status: "completed" }, sequence_number: 6 },
      ])
    );
    const { result } = renderHook(() => useResponsesChat());
    await act(async () => { await result.current.resumeIfInterrupted(); });
    expect(responsesApi.streamResume).toHaveBeenCalledWith("resp_5", 4, expect.anything());
    const st = useChatStore.getState();
    expect(st.messages[1].text).toBe("partial");
    expect(st.messages[1].status).toBe("completed");
    expect(st.lastResponseId).toBe("resp_5");
  });

  it("resumeIfInterrupted is a no-op when the last message is not streaming", async () => {
    useChatStore.setState({
      messages: [{ id: "a", role: "assistant", text: "done", reasoning: "", reasoningStatus: "idle", status: "completed", responseId: "r" }],
    });
    const { result } = renderHook(() => useResponsesChat());
    await act(async () => { await result.current.resumeIfInterrupted(); });
    expect(responsesApi.streamResume).not.toHaveBeenCalled();
  });
});
