import { describe, it, expect, beforeEach } from "vitest";
import { useChatStore, normalizeHistoryMessages } from "../chat";
import type { ConversationDetail } from "../../types";

beforeEach(() => useChatStore.getState().reset());

describe("chat store", () => {
  it("appends and patches the last message", () => {
    const s = useChatStore.getState();
    s.appendMessage({
      id: "u1",
      role: "user",
      text: "hi",
      reasoning: "",
      reasoningStatus: "idle",
      status: "completed",
      toolCalls: [],
    });
    s.appendMessage({
      id: "a1",
      role: "assistant",
      text: "",
      reasoning: "",
      reasoningStatus: "idle",
      status: "streaming",
      toolCalls: [],
    });
    useChatStore.getState().updateLast({ text: "hello", status: "completed" });
    const msgs = useChatStore.getState().messages;
    expect(msgs).toHaveLength(2);
    expect(msgs[1].text).toBe("hello");
    expect(msgs[1].status).toBe("completed");
  });

  it("sets anchors and model", () => {
    useChatStore.getState().setAnchors({ conversationId: "c1", lastResponseId: "r1" });
    useChatStore.getState().setModel("gpt-4o");
    const st = useChatStore.getState();
    expect(st.conversationId).toBe("c1");
    expect(st.lastResponseId).toBe("r1");
    expect(st.model).toBe("gpt-4o");
  });

  it("loadHistory replaces messages and sets anchors", () => {
    const detail: ConversationDetail = {
      id: "c9",
      title: "t",
      created_at: null,
      updated_at: null,
      latest_response_id: "r9",
      messages: [
        { role: "user", text: "q", response_id: "r9" } as never,
        {
          role: "assistant",
          text: "a",
          reasoning: "why",
          response_id: "r9",
          previous_response_id: null,
          status: "completed",
        } as never,
      ],
    };
    useChatStore.getState().loadHistory(detail);
    const st = useChatStore.getState();
    expect(st.conversationId).toBe("c9");
    expect(st.lastResponseId).toBe("r9");
    expect(st.messages).toHaveLength(2);
    expect(st.messages[1].reasoning).toBe("why");
    expect(st.messages[1].reasoningStatus).toBe("done");
  });
});

describe("normalizeHistoryMessages", () => {
  it("normalizeHistoryMessages maps tool_calls", () => {
    const out = normalizeHistoryMessages({
      id: "c", title: null, created_at: null, updated_at: null, latest_response_id: "r1",
      messages: [{ role: "assistant", text: "a", reasoning: "", response_id: "r1",
        previous_response_id: null, status: "completed",
        tool_calls: [{ call_id: "c1", name: "web_fetch", arguments: "{}", output: "PAGE" }] } as never],
    });
    expect(out[0].toolCalls).toEqual([
      { id: "c1", name: "web_fetch", arguments: "{}", status: "done", output: "PAGE" },
    ]);
  });

  it("reconstructs a persisted HITL notice onto the tool call", () => {
    const notice = { kind: "aliyun_authorization", bound: false, interrupt: true };
    const out = normalizeHistoryMessages({
      id: "c", title: null, created_at: null, updated_at: null, latest_response_id: "r1",
      messages: [{ role: "assistant", text: "已暂停", reasoning: "", response_id: "r1",
        previous_response_id: null, status: "completed",
        tool_calls: [{ call_id: "c1", name: "shell", arguments: "{}", output: "denied", notice }] } as never],
    });
    expect(out[0].toolCalls[0].notice).toEqual(notice);
  });

  it("fills ids, camelCases, and sets reasoningStatus", () => {
    const out = normalizeHistoryMessages({
      id: "c",
      title: null,
      created_at: null,
      updated_at: null,
      latest_response_id: "r1",
      messages: [
        {
          role: "assistant",
          text: "a",
          reasoning: "",
          response_id: "r1",
          previous_response_id: "r0",
          status: "completed",
        } as never,
      ],
    });
    expect(out[0].id).toBe("r1");
    expect(out[0].responseId).toBe("r1");
    expect(out[0].previousResponseId).toBe("r0");
    expect(out[0].reasoningStatus).toBe("idle"); // empty reasoning -> idle
  });
});
