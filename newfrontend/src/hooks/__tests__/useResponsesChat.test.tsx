import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

vi.mock("../../api/client", () => ({ streamResponse: vi.fn() }));
vi.mock("../../lib/user", () => ({ getUserId: () => "u1" }));
vi.mock("../../store/conversations", () => ({
  useConversationsStore: { getState: () => ({ refresh: vi.fn() }) },
}));

import { useResponsesChat } from "../useResponsesChat";
import { useChatStore } from "../../store/chat";
import * as client from "../../api/client";

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

describe("useResponsesChat", () => {
  it("send streams a turn, captures text and advances anchors", async () => {
    (client.streamResponse as any).mockReturnValue(
      streamOf([
        { type: "response.created", response: { id: "resp_1", conversation: { id: "conv_1" } } },
        { type: "response.output_text.delta", delta: "hi" },
        {
          type: "response.completed",
          response: {
            id: "resp_1",
            conversation: { id: "conv_1" },
            status: "completed",
            usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
          },
        },
      ])
    );
    const { result } = renderHook(() => useResponsesChat());
    await act(async () => {
      await result.current.send("hello");
    });
    const st = useChatStore.getState();
    expect(st.messages.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(st.messages[1].text).toBe("hi");
    expect(st.messages[1].status).toBe("completed");
    expect(st.conversationId).toBe("conv_1");
    expect(st.lastResponseId).toBe("resp_1");
    // continuation: a second send carries the conversation anchor
    (client.streamResponse as any).mockReturnValue(
      streamOf([
        { type: "response.created", response: { id: "resp_2", conversation: { id: "conv_1" } } },
        { type: "response.completed", response: { id: "resp_2", conversation: { id: "conv_1" }, status: "completed" } },
      ])
    );
    await act(async () => {
      await result.current.send("again");
    });
    const params = (client.streamResponse as any).mock.calls[1][0];
    expect(params.conversation).toBe("conv_1");
    expect(params.previous_response_id).toBe("resp_1");
    expect(params.user_id).toBe("u1");
  });

  it("stop aborts and marks the bubble stopped without advancing anchors", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    (client.streamResponse as any).mockReturnValue({
      async *[Symbol.asyncIterator]() {
        yield { type: "response.created", response: { id: "resp_1", conversation: { id: "conv_1" } } };
        yield { type: "response.output_text.delta", delta: "partial" };
        await gate; // never resolves before stop()
      },
    });
    const { result } = renderHook(() => useResponsesChat());
    let sendPromise: Promise<void>;
    await act(async () => {
      sendPromise = result.current.send("hello");
      await Promise.resolve();
    });
    act(() => result.current.stop());
    release();
    await act(async () => {
      await sendPromise;
    });
    const st = useChatStore.getState();
    expect(st.messages[1].status).toBe("stopped");
    expect(st.messages[1].text).toBe("partial");
    // Stop does NOT advance anchors (first turn -> next send starts fresh)
    expect(st.conversationId).toBeUndefined();
    expect(st.lastResponseId).toBeUndefined();
  });
});
