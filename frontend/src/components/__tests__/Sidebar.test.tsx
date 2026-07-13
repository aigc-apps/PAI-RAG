import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router-dom";

vi.mock("../../api/conversations", () => ({
  getConversation: vi.fn(),
  listConversations: vi.fn().mockResolvedValue([]),
  deleteConversation: vi.fn(),
}));
// UserMenu probes the network on mount; it's irrelevant to the sidebar list.
vi.mock("../UserMenu", () => ({ UserMenu: () => null }));

import { Sidebar } from "../Sidebar";
import { useChatStore } from "../../store/chat";
import { useConversationsStore } from "../../store/conversations";
import type { ChatMessage } from "../../types";

const userMsg = (text: string): ChatMessage => ({
  id: "u", role: "user", text, reasoning: "", reasoningStatus: "idle",
  status: "completed", toolCalls: [],
});
const streamingAssistant = (): ChatMessage => ({
  id: "a", role: "assistant", text: "partial", reasoning: "", reasoningStatus: "idle",
  status: "streaming", responseId: "resp_1", toolCalls: [],
});

function renderSidebar(path = "/") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Sidebar />
      <div data-testid="location">{<Location />}</div>
    </MemoryRouter>,
  );
}

function Location() {
  return <>{useLocation().pathname}</>;
}

beforeEach(() => {
  vi.clearAllMocks();
  useChatStore.getState().reset();
  // No-op refresh so tests own `items` (the real one would async-clobber it).
  useConversationsStore.setState({
    items: [],
    selectedId: undefined,
    refresh: vi.fn().mockResolvedValue(undefined),
  });
});

describe("Sidebar", () => {
  it("shows a row for a brand-new draft immediately", () => {
    // reset() installs one fresh empty draft as the active runtime.
    renderSidebar();
    expect(screen.getByText("新对话")).toBeInTheDocument();
    expect(screen.queryByText("暂无对话")).not.toBeInTheDocument();
  });

  it("shows a streaming conversation before the server list has it", () => {
    const key = useChatStore.getState().activeKey;
    useChatStore.getState().appendMessage(key, userMsg("streaming question"));
    useChatStore.getState().appendMessage(key, streamingAssistant());
    useChatStore.getState().setStatusOf(key, "streaming");
    useChatStore.getState().setAnchorsOf(key, { conversationId: "conv_live" });
    // Server list is still empty (refresh happens on completion).
    renderSidebar();
    expect(screen.getByText("streaming question")).toBeInTheDocument();
    expect(screen.getByLabelText("生成中")).toBeInTheDocument();
  });

  it("clicking an overlay row activates its runtime and can switch back mid-stream", async () => {
    const user = userEvent.setup();
    // A: a streaming conversation not yet in the server list.
    const keyA = useChatStore.getState().activeKey;
    useChatStore.getState().appendMessage(keyA, userMsg("conversation A"));
    useChatStore.getState().appendMessage(keyA, streamingAssistant());
    useChatStore.getState().setStatusOf(keyA, "streaming");
    // Switch to a fresh draft B (as "新建对话" does).
    const keyB = useChatStore.getState().newDraft();
    expect(useChatStore.getState().activeKey).toBe(keyB);

    renderSidebar();
    // A is still listed and switchable back to.
    await user.click(screen.getByText("conversation A"));
    expect(useChatStore.getState().activeKey).toBe(keyA);
  });

  it("does not duplicate a conversation already in the server list", () => {
    // Open a saved conversation: its runtime carries a conversationId that IS in
    // the server list, so it renders once (from the list), not twice.
    const key = useChatStore.getState().activeKey;
    useChatStore.getState().appendMessage(key, userMsg("saved chat"));
    useChatStore.getState().setAnchorsOf(key, { conversationId: "conv_saved" });
    useChatStore.getState().setStatusOf(key, "idle");
    useConversationsStore.setState({
      items: [{ id: "conv_saved", title: "Saved chat", created_at: null, updated_at: null, last_response_id: null }],
    });
    renderSidebar();
    expect(screen.getAllByText(/saved chat/i)).toHaveLength(1);
  });

  it("navigates persisted rows to their conversation route", async () => {
    const user = userEvent.setup();
    useConversationsStore.setState({
      items: [{ id: "c1", title: "Saved route", created_at: null, updated_at: null, last_response_id: null }],
    });
    renderSidebar();

    await user.click(screen.getByText("Saved route"));

    expect(screen.getByTestId("location")).toHaveTextContent("/chat/c1");
  });

  it("navigates New chat to the root route", async () => {
    const user = userEvent.setup();
    renderSidebar("/chat/c1");

    await user.click(screen.getByRole("button", { name: /new chat|新建对话/i }));

    expect(screen.getByTestId("location")).toHaveTextContent("/");
  });
});
