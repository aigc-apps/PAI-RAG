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
  getAgentConfig: vi.fn(),
  saveSetup: vi.fn(),
  saveAgentConfig: vi.fn(),
  getAliyunStatus: vi.fn().mockResolvedValue({
    configured: false, bound: false, region: "", external_id: null,
  }),
}));

import { App } from "../App";
import { useChatStore } from "../../store/chat";
import { useAgentConfigStore } from "../../store/agentConfig";
import { useAuthStore } from "../../store/auth";

function setAuth(over: Partial<ReturnType<typeof useAuthStore.getState>>) {
  useAuthStore.setState({
    user: null,
    phase: "anonymous",
    isAdmin: false,
    bootstrapNeeded: false,
    hydrate: async () => {},
    ...over,
  });
}

beforeEach(() => {
  useChatStore.getState().reset();
  useAgentConfigStore.setState({ doc: undefined, loading: false, error: undefined });
});

describe("App auth guards", () => {
  it("shows Create-Admin when bootstrap is needed", async () => {
    setAuth({ bootstrapNeeded: true });
    render(<App />);
    expect(await screen.findByText(/create admin account/i)).toBeInTheDocument();
  });

  it("shows Login when anonymous", async () => {
    setAuth({ phase: "anonymous" });
    render(<App />);
    expect(await screen.findByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("hides the settings gear for a regular user", async () => {
    setAuth({
      phase: "authenticated",
      isAdmin: false,
      user: { id: "u1", email: "u@b.com", role: "user", status: "active", display_name: null },
    });
    render(<App />);
    // Chat surface is up…
    expect(await screen.findByRole("textbox")).toBeInTheDocument();
    // …but no admin control.
    expect(screen.queryByRole("button", { name: /open settings/i })).toBeNull();
  });

  it("shows the settings gear for an admin", async () => {
    setAuth({
      phase: "authenticated",
      isAdmin: true,
      user: { id: "a1", email: "a@b.com", role: "admin", status: "active", display_name: null },
    });
    render(<App />);
    expect(
      await screen.findByRole("button", { name: /open settings/i })
    ).toBeInTheDocument();
  });
});
