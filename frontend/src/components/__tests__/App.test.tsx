import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router-dom";

vi.mock("../../api/conversations", () => ({
  listConversations: vi.fn().mockResolvedValue([]),
  getConversation: vi.fn(),
  deleteConversation: vi.fn(),
}));
vi.mock("../../api/client", () => ({ streamResponse: vi.fn() }));
vi.mock("../KnowledgeView", () => ({
  KnowledgeView: () => <div>Knowledge manager opened</div>,
}));
vi.mock("../UsersView", () => ({
  UsersView: () => <div>User management opened</div>,
}));
vi.mock("../../api/models", () => ({
  listModels: vi.fn().mockResolvedValue({ ids: [], default: "", models: [], defaultEmbedding: null, defaultRerank: null }),
}));
vi.mock("../../api/agentConfig", () => ({
  getSetup: vi.fn().mockResolvedValue({
    setup: { completed: true, skipped_steps: [] },
    models: {},
    knowledgebase: { vectordb: { engine: "local", url: "", index_prefix: "kb", api_key: "", api_key_env: "", username: "", password: "", password_env: "", verify_certs: true, timeout: 30, status: "healthy", secret_configured: false } },
    skills: { root: "./data/skills", mount: { mount_root: "/mnt/skills" } },
    default_agent: "main",
    agents: [],
    providers: [],
    capabilities: [],
  }),
  getAgentConfig: vi.fn().mockResolvedValue({
    setup: { completed: true, skipped_steps: [] },
    models: {},
    knowledgebase: { vectordb: { engine: "local", url: "", index_prefix: "kb", api_key: "", api_key_env: "", username: "", password: "", password_env: "", verify_certs: true, timeout: 30, status: "healthy", secret_configured: false } },
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
import { getConversation } from "../../api/conversations";
import { useChatStore } from "../../store/chat";
import { useConversationsStore } from "../../store/conversations";
import { useAgentConfigStore } from "../../store/agentConfig";
import { useAuthStore } from "../../store/auth";
import { useI18nStore } from "../../i18n";

function renderApp(path = "/") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
      <LocationProbe />
    </MemoryRouter>,
  );
}

function LocationProbe() {
  return <div data-testid="location">{useLocation().pathname}</div>;
}

beforeEach(() => {
  useI18nStore.getState().setLang("en");
  useChatStore.getState().reset();
  useConversationsStore.setState({ items: [], selectedId: undefined });
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
    renderApp();
    expect(await screen.findByRole("textbox")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /new chat/i })
    ).toBeInTheDocument();
  });

  it("renders existing messages from the chat store", async () => {
    const key = useChatStore.getState().activeKey;
    useChatStore.getState().appendMessage(key, {
      id: "u",
      role: "user",
      text: "hello there",
      reasoning: "",
      reasoningStatus: "idle",
      status: "completed",
      toolCalls: [],
    });
    renderApp();
    // Appears both in the message list and as the sidebar row title for the
    // active runtime — assert it rendered at least once.
    expect((await screen.findAllByText("hello there")).length).toBeGreaterThan(0);
  });

  it("opens knowledge management from settings", async () => {
    const user = userEvent.setup();
    renderApp();

    await user.click(await screen.findByRole("button", { name: /account menu/i }));
    await user.click(await screen.findByText(/^settings$/i));
    await user.click(await screen.findByRole("button", { name: "Knowledge" }));

    expect(await screen.findByText("Knowledge manager opened")).toBeInTheDocument();
  });

  it.each([
    ["/users", "User management opened"],
    ["/knowledge", "Knowledge manager opened"],
  ])("renders the admin surface for %s", async (path, text) => {
    renderApp(path);
    expect(await screen.findByText(text)).toBeInTheDocument();
  });

  it("renders settings for its direct route", async () => {
    renderApp("/settings/agents");
    expect(await screen.findByRole("heading", { name: "Settings" })).toBeInTheDocument();
  });

  it("redirects an unknown route to chat", async () => {
    renderApp("/not-a-real-page");
    expect(await screen.findByRole("textbox")).toBeInTheDocument();
  });

  it("hydrates a conversation selected by the chat route", async () => {
    vi.mocked(getConversation).mockResolvedValue({
      id: "c1",
      title: "Saved",
      created_at: null,
      updated_at: null,
      latest_response_id: "r1",
      messages: [{
        id: "r1:u:0",
        role: "user",
        text: "loaded from route",
        reasoning: "",
        reasoningStatus: "idle",
        status: "completed",
        responseId: "r1",
        toolCalls: [],
      }],
    });

    renderApp("/chat/c1");

    expect((await screen.findAllByText("loaded from route")).length).toBeGreaterThan(0);
    expect(useConversationsStore.getState().selectedId).toBe("c1");
  });

  it("recovers an invalid chat route back to a fresh chat", async () => {
    vi.mocked(getConversation).mockRejectedValue(new Error("missing"));

    renderApp("/chat/missing");

    await screen.findByRole("textbox");
    await waitFor(() => {
      expect(screen.getByTestId("location").textContent).toBe("/");
    });
  });

  it("deep-links settings sections and updates the URL on tab changes", async () => {
    const user = userEvent.setup();
    renderApp("/settings/connections");

    expect(await screen.findByRole("heading", { name: "Models", level: 2 })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Capabilities" }));
    expect(screen.getByTestId("location").textContent).toBe("/settings/tools");
  });

  it("canonicalizes missing and invalid settings sections", async () => {
    renderApp("/settings/not-a-section");

    await waitFor(() => {
      expect(screen.getByTestId("location").textContent).toBe("/settings/agents");
    });
  });

  it("redirects legacy knowledge deep-links to the settings tab", async () => {
    renderApp("/knowledge/kb_1/not-a-tab");

    await waitFor(() => {
      expect(screen.getByTestId("location").textContent).toBe("/settings/knowledge");
    });
    expect(await screen.findByText("Knowledge manager opened")).toBeInTheDocument();
  });
});
