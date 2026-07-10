import { describe, it, expect, beforeEach } from "vitest";
import { activeRuntime, useChatStore, normalizeHistoryMessages } from "../chat";
import type { ChatMessage, ConversationDetail } from "../../types";

beforeEach(() => useChatStore.getState().reset());

const active = () => activeRuntime(useChatStore.getState())!;
const userMsg = (id: string, text: string): ChatMessage => ({
  id, role: "user", text, reasoning: "", reasoningStatus: "idle",
  status: "completed", toolCalls: [],
});
const assistantMsg = (id: string): ChatMessage => ({
  id, role: "assistant", text: "", reasoning: "", reasoningStatus: "idle",
  status: "streaming", toolCalls: [],
});

describe("chat store", () => {
  it("appends and patches the last message of the active runtime", () => {
    const key = useChatStore.getState().activeKey;
    useChatStore.getState().appendMessage(key, userMsg("u1", "hi"));
    useChatStore.getState().appendMessage(key, assistantMsg("a1"));
    useChatStore.getState().updateLastOf(key, { text: "hello", status: "completed" });
    const msgs = active().messages;
    expect(msgs).toHaveLength(2);
    expect(msgs[1].text).toBe("hello");
    expect(msgs[1].status).toBe("completed");
  });

  it("sets anchors on the runtime and model globally", () => {
    const key = useChatStore.getState().activeKey;
    useChatStore.getState().setAnchorsOf(key, { conversationId: "c1", lastResponseId: "r1" });
    useChatStore.getState().setModel("gpt-4o");
    expect(active().conversationId).toBe("c1");
    expect(active().lastResponseId).toBe("r1");
    expect(useChatStore.getState().model).toBe("gpt-4o");
  });

  it("isolates writes per runtime — a background loop never touches the active one", () => {
    const a = useChatStore.getState().activeKey;
    useChatStore.getState().appendMessage(a, assistantMsg("a1"));
    // Open a fresh conversation; `a` is now a background runtime.
    const b = useChatStore.getState().newDraft();
    expect(useChatStore.getState().activeKey).toBe(b);
    useChatStore.getState().appendMessage(b, assistantMsg("b1"));
    // A late write addressed to `a` lands in a's slice, not the visible one.
    useChatStore.getState().updateLastOf(a, { text: "from A" });
    expect(useChatStore.getState().runtimes[a].messages[0].text).toBe("from A");
    expect(active().messages[0].text).toBe(""); // b untouched
  });

  it("newDraft prunes the previous empty draft but keeps ones with messages", () => {
    const empty = useChatStore.getState().activeKey; // empty + idle
    useChatStore.getState().newDraft();
    expect(useChatStore.getState().runtimes[empty]).toBeUndefined();

    const withMsg = useChatStore.getState().activeKey;
    useChatStore.getState().appendMessage(withMsg, userMsg("u", "keep me"));
    useChatStore.getState().newDraft();
    expect(useChatStore.getState().runtimes[withMsg]).toBeDefined();
  });

  it("hydrate loads history into a runtime and activates it", () => {
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
    useChatStore.getState().hydrate(detail);
    const rt = active();
    expect(rt.conversationId).toBe("c9");
    expect(rt.lastResponseId).toBe("r9");
    expect(rt.messages).toHaveLength(2);
    expect(rt.messages[1].reasoning).toBe("why");
    expect(rt.messages[1].reasoningStatus).toBe("done");
  });

  it("activateByConversationId switches to an existing runtime; dropByConversationId removes it", () => {
    useChatStore.getState().hydrate({
      id: "c1", title: null, created_at: null, updated_at: null,
      latest_response_id: "r1", messages: [],
    });
    const first = useChatStore.getState().activeKey;
    useChatStore.getState().newDraft(); // move active away
    expect(useChatStore.getState().activateByConversationId("c1")).toBe(true);
    expect(useChatStore.getState().activeKey).toBe(first);
    expect(useChatStore.getState().activateByConversationId("nope")).toBe(false);

    // Deleting the active conversation installs a fresh empty draft.
    useChatStore.getState().dropByConversationId("c1");
    expect(useChatStore.getState().runtimes[first]).toBeUndefined();
    expect(active().messages).toHaveLength(0);
    expect(active().conversationId).toBeUndefined();
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
