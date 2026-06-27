import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("../../api/conversations", () => ({
  listConversations: vi.fn().mockResolvedValue([]),
  getConversation: vi.fn(),
  deleteConversation: vi.fn(),
}));
vi.mock("../../api/client", () => ({ streamResponse: vi.fn() }));
vi.mock("../../lib/user", () => ({ getUserId: () => "u1" }));

import { App } from "../App";
import { useChatStore } from "../../store/chat";

beforeEach(() => useChatStore.getState().reset());

describe("App", () => {
  it("renders the composer and a New chat control", async () => {
    render(<App />);
    expect(screen.getByRole("textbox")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /new chat/i })
    ).toBeInTheDocument();
  });

  it("renders existing messages from the chat store", () => {
    useChatStore.setState({
      messages: [
        {
          id: "u",
          role: "user",
          text: "hello there",
          reasoning: "",
          reasoningStatus: "idle",
          status: "completed",
          toolCalls: [],
        },
      ],
    });
    render(<App />);
    expect(screen.getByText("hello there")).toBeInTheDocument();
  });
});
