import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

vi.mock("../../api/conversations", () => ({
  listConversations: vi.fn().mockResolvedValue([]),
  getConversation: vi.fn(),
  deleteConversation: vi.fn(),
}));
vi.mock("../../api/client", () => ({ streamResponse: vi.fn() }));
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
import { useI18nStore } from "../../i18n";

function renderApp(path = "/") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

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
  useI18nStore.getState().setLang("en");
  useChatStore.getState().reset();
  useAgentConfigStore.setState({ doc: undefined, loading: false, error: undefined });
});

describe("App auth guards", () => {
  it("shows Create-Admin when bootstrap is needed", async () => {
    setAuth({ bootstrapNeeded: true });
    renderApp();
    expect(await screen.findByText(/create admin account/i)).toBeInTheDocument();
  });

  it("shows Login when anonymous", async () => {
    setAuth({ phase: "anonymous" });
    renderApp();
    expect(await screen.findByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("hides Settings inside the account menu for a regular user", async () => {
    setAuth({
      phase: "authenticated",
      isAdmin: false,
      user: { id: "u1", email: "u@b.com", role: "user", status: "active", display_name: null },
    });
    renderApp();
    // Chat surface is up…
    expect(await screen.findByRole("textbox")).toBeInTheDocument();
    // …open the bottom-left account menu — a non-admin sees no Settings item.
    fireEvent.click(await screen.findByRole("button", { name: /account menu/i }));
    await screen.findByText(/sign out/i);
    expect(screen.queryByText(/^settings$/i)).toBeNull();
  });

  it("shows Settings inside the account menu for an admin", async () => {
    setAuth({
      phase: "authenticated",
      isAdmin: true,
      user: { id: "a1", email: "a@b.com", role: "admin", status: "active", display_name: null },
    });
    renderApp();
    fireEvent.click(await screen.findByRole("button", { name: /account menu/i }));
    expect(await screen.findByText(/^settings$/i)).toBeInTheDocument();
  });

  it("redirects a regular user away from admin routes", async () => {
    setAuth({
      phase: "authenticated",
      isAdmin: false,
      user: { id: "u1", email: "u@b.com", role: "user", status: "active", display_name: null },
    });
    renderApp("/users");
    expect(await screen.findByRole("textbox")).toBeInTheDocument();
  });
});
