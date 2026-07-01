import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("../../api/conversations", () => ({
  listConversations: vi.fn().mockResolvedValue([]),
  getConversation: vi.fn(),
  deleteConversation: vi.fn(),
}));
vi.mock("../../api/client", () => ({ streamResponse: vi.fn() }));
vi.mock("../../lib/user", () => ({ getUserId: () => "u1" }));
vi.mock("../../api/models", () => ({
  listModels: vi.fn().mockResolvedValue({ ids: [], default: "" }),
}));
vi.mock("../../api/agentConfig", () => ({
  getSetup: vi.fn().mockResolvedValue({
    setup: { completed: true, skipped_steps: [] },
    models: {},
    skills: { root: "./data/skills", mount: { mount_root: "/mnt/skills" } },
    default_agent: "main",
    agents: [],
    providers: [],
    capabilities: [],
  }),
  getAgentConfig: vi.fn().mockResolvedValue({
    setup: { completed: true, skipped_steps: [] },
    models: {},
    skills: { root: "./data/skills", mount: { mount_root: "/mnt/skills" } },
    default_agent: "main",
    agents: [],
    providers: [],
    capabilities: [],
  }),
  saveSetup: vi.fn(),
  saveAgentConfig: vi.fn(),
}));

import { App } from "../App";
import { useChatStore } from "../../store/chat";
import { useAgentConfigStore } from "../../store/agentConfig";

beforeEach(() => {
  useChatStore.getState().reset();
  useAgentConfigStore.setState({ doc: undefined, loading: false, error: undefined });
});

describe("App", () => {
  it("renders the composer and a New chat control", async () => {
    render(<App />);
    expect(await screen.findByRole("textbox")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /new chat/i })
    ).toBeInTheDocument();
  });

  it("renders existing messages from the chat store", async () => {
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
    expect(await screen.findByText("hello there")).toBeInTheDocument();
  });
});
