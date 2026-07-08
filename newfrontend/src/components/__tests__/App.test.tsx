import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("../../api/conversations", () => ({
  listConversations: vi.fn().mockResolvedValue([]),
  getConversation: vi.fn(),
  deleteConversation: vi.fn(),
}));
vi.mock("../../api/client", () => ({ streamResponse: vi.fn() }));
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
  // UserMenu probes this on mount.
  getAliyunStatus: vi.fn().mockResolvedValue({
    configured: false, bound: false, region: "", external_id: null,
  }),
}));

import { App } from "../App";
import { useChatStore } from "../../store/chat";
import { useAgentConfigStore } from "../../store/agentConfig";
import { useAuthStore } from "../../store/auth";

beforeEach(() => {
  useChatStore.getState().reset();
  useAgentConfigStore.setState({ doc: undefined, loading: false, error: undefined });
  // Start authenticated as an admin with a no-op hydrate so the guards fall
  // straight through to the chat surface.
  useAuthStore.setState({
    user: { id: "u1", email: "admin@example.com", role: "admin", status: "active", display_name: null },
    phase: "authenticated",
    isAdmin: true,
    bootstrapNeeded: false,
    hydrate: async () => {},
  });
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
