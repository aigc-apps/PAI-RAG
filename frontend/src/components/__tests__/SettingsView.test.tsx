import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SettingsView } from "../SettingsView";
import { useAgentConfigStore } from "../../store/agentConfig";
import type { AgentConfigDocument } from "../../api/agentConfig";

const generateCodeManifest = vi.fn();
vi.mock("../../api/agentConfig", async (importActual) => ({
  ...(await importActual<typeof import("../../api/agentConfig")>()),
  generateCodeManifest: (agentId: string) => generateCodeManifest(agentId),
}));

const listKnowledgeBases = vi.fn(async () => [] as unknown[]);
vi.mock("../../api/knowledge", async (importActual) => ({
  ...(await importActual<typeof import("../../api/knowledge")>()),
  listKnowledgeBases: () => listKnowledgeBases(),
}));

const baseDoc: AgentConfigDocument = {
  setup: { completed: true, skipped_steps: [] },
  models: {},
  default_instructions: "",
  knowledgebase: {
    vectordb: {
      engine: "local",
      url: "",
      index_prefix: "kb",
      api_key: "",
      api_key_env: "",
      username: "",
      password: "",
      password_env: "",
      verify_certs: true,
      timeout: 30,
      status: "healthy",
      secret_configured: false,
    },
  },
  skills: { root: "./data/skills", mount: { mount_root: "/mnt/skills" } },
  default_agent: "main",
  agents: [
    {
      id: "main",
      name: "Main",
      description: "",
      model: "",
      instructions: "",
      knowledge: { kb_ids: [] },
      code_manifest: "",
      tools: { include: ["current_datetime"], exclude: ["code_sandbox"] },
      skills: { enabled: [] },
      settings: {},
    },
  ],
  providers: [
    {
      id: "sandbox.default",
      type: "sandbox",
      name: "Sandbox runtime",
      status: "missing_config",
      settings: {
        provider: "agentrun_rest",
        endpoint: "",
        api_key_env: "AGENTRUN_SANDBOX_API_KEY",
        api_key_header: "X-API-Key",
        account_id_env: "AGENTRUN_ACCOUNT_ID",
        template_name: "",
        template_type: "CodeInterpreter",
        isolation_scope: "conversation",
        idle_timeout_seconds: 600,
        session_idle_seconds: 600,
        timeout_seconds: 30,
        cwd: "/home/user",
        oss_mount_config: { mount_points: [] },
        nas_config: { mount_points: [] },
      },
      secret_configured: false,
      used_by: ["sandbox"],
    },
  ],
  capabilities: [
    {
      id: "knowledge",
      kind: "core_tool",
      name: "Knowledge Base",
      description: "Retrieve from local documents.",
      enabled: true,
      permission: "auto",
      status: "ready",
      dependencies: [],
      provider_refs: [],
      settings: { mode: "local" },
    },
    {
      id: "sandbox",
      kind: "core_tool",
      name: "Sandbox",
      description: "Run code in a sandbox.",
      enabled: false,
      permission: "disabled",
      status: "disabled",
      dependencies: [],
      provider_refs: ["sandbox.default"],
      settings: {},
    },
  ],
};

// The sandbox capability turned on globally.
const sandboxCapDoc: AgentConfigDocument = {
  ...baseDoc,
  capabilities: baseDoc.capabilities.map((c) =>
    c.id === "sandbox" ? { ...c, enabled: true, permission: "auto", status: "ready" } : c
  ),
};

// ...and the selected agent has actually activated code browsing (code_sandbox in
// its own toolbox). Only then does the manifest section belong to the agent.
const codeBrowsingDoc: AgentConfigDocument = {
  ...sandboxCapDoc,
  agents: sandboxCapDoc.agents.map((a) =>
    a.id === "main"
      ? { ...a, tools: { include: [...a.tools.include, "code_sandbox"], exclude: [] } }
      : a
  ),
};

describe("SettingsView", () => {
  beforeEach(() => {
    generateCodeManifest.mockReset();
    listKnowledgeBases.mockReset();
    listKnowledgeBases.mockResolvedValue([]);
    useAgentConfigStore.setState({
      doc: baseDoc,
      loading: false,
      error: undefined,
      save: vi.fn(async (doc: AgentConfigDocument) => doc),
    });
  });

  it("configures sandbox REST provider settings", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });

    render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Tools" }));
    await user.click(screen.getByRole("button", { name: "Configure" }));

    await user.type(
      screen.getByLabelText("Sandbox gateway endpoint"),
      "https://sandbox-gateway.internal"
    );
    await user.type(screen.getByLabelText("Sandbox template name"), "code-template");
    await user.clear(screen.getByLabelText("Sandbox account ID env"));
    await user.type(screen.getByLabelText("Sandbox account ID env"), "ALIYUN_ACCOUNT_ID");
    await user.clear(screen.getByLabelText("Sandbox session idle seconds"));
    await user.type(screen.getByLabelText("Sandbox session idle seconds"), "900");

    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    const provider = saved.providers.find((item) => item.id === "sandbox.default");
    const sandbox = saved.capabilities.find((item) => item.id === "sandbox");
    const agent = saved.agents[0];

    expect(provider?.settings.provider).toBe("agentrun_rest");
    expect(provider?.settings.endpoint).toBe("https://sandbox-gateway.internal");
    expect(provider?.settings.template_name).toBe("code-template");
    expect(provider?.settings.api_key_header).toBe("X-API-Key");
    expect(provider?.settings.account_id_env).toBe("ALIYUN_ACCOUNT_ID");
    expect(provider?.settings.isolation_scope).toBe("conversation");
    expect(provider?.settings.template_type).toBe("CodeInterpreter");
    expect(provider?.settings.timeout_seconds).toBe(30);
    expect(provider?.settings.create_path).toBe("/sandboxes");
    expect(provider?.settings.execute_path).toBe("/sandboxes/{sandbox_id}/contexts/execute");
    expect(provider?.settings.stop_path).toBe("/sandboxes/{sandbox_id}/stop");
    expect(provider?.settings.wake_path).toBeUndefined();
    expect(provider?.settings.session_idle_seconds).toBe(900);
    expect(provider?.settings.oss_mount_config).toEqual({ mount_points: [] });
    expect(sandbox?.enabled).toBe(true);
    expect(sandbox?.permission).toBe("auto");
    expect(agent.tools.include).toContain("code_sandbox");
    expect(agent.tools.exclude).not.toContain("code_sandbox");
  });

  it("saves with endpoint empty (auto-derived by backend)", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });

    render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Tools" }));
    await user.click(screen.getByRole("button", { name: "Configure" }));

    // Leave gateway endpoint empty — backend auto-derives from account id.
    await user.type(screen.getByLabelText("Sandbox template name"), "code-template");
    // api_key_env + account_id_env are pre-filled in baseDoc, satisfying the
    // required trio (template_name + api_key + account_id) without an endpoint.

    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    const provider = saved.providers.find((item) => item.id === "sandbox.default");
    expect(provider?.settings.endpoint).toBe("");
    expect(provider?.settings.template_name).toBe("code-template");
  });

  it("blocks save when api key and env are both missing", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });

    render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Tools" }));
    await user.click(screen.getByRole("button", { name: "Configure" }));

    await user.type(screen.getByLabelText("Sandbox template name"), "code-template");
    await user.clear(screen.getByLabelText("Sandbox gateway API key env"));

    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(save).not.toHaveBeenCalled();
    expect(screen.getByText("Template name, API key, and account id are required")).toBeInTheDocument();
  });

  it("configures the global elasticsearch vector database", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });

    render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Tools" }));
    await user.click(screen.getByRole("button", { name: "Vector DB" }));

    // Elasticsearch is the only engine — its fields show immediately, no chooser.
    expect(screen.queryByLabelText("Engine")).not.toBeInTheDocument();
    await user.type(screen.getByLabelText("URL"), "https://es:9200");
    await user.type(screen.getByLabelText("API Key"), "es-secret");

    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    expect(saved.knowledgebase.vectordb.engine).toBe("elasticsearch");
    expect(saved.knowledgebase.vectordb.url).toBe("https://es:9200");
    expect(saved.knowledgebase.vectordb.api_key).toBe("es-secret");
  });

  it("hides the code manifest section unless the agent activated code browsing", () => {
    // Sandbox is enabled globally, but the selected agent has code_sandbox excluded
    // -> the manifest is not one of its attributes, so the section stays hidden.
    render(<SettingsView doc={sandboxCapDoc} onBack={vi.fn()} />);
    expect(screen.queryByText("代码库配置单")).not.toBeInTheDocument();
  });

  it("generates a code manifest and saves it onto the agent", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });
    generateCodeManifest.mockResolvedValue({ manifest: "- repo-a — the API server" });

    render(<SettingsView doc={codeBrowsingDoc} onBack={vi.fn()} />);

    // The manifest lives in the Tools dialog (it belongs with the code-sandbox tool).
    await user.click(screen.getByRole("button", { name: "编辑工具" }));
    // Section is visible once this agent has activated code browsing.
    expect(screen.getByText("代码库配置单")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /AI 生成/ }));

    expect(generateCodeManifest).toHaveBeenCalledWith("main");
    // Generated text lands in the textarea...
    expect(await screen.findByDisplayValue("- repo-a — the API server")).toBeInTheDocument();
    // ...and is committed via the whole-doc save.
    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    expect(saved.agents[0].code_manifest).toBe("- repo-a — the API server");
  });

  it("edits the agent Persona in a dialog and commits on 保存", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });

    render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);

    // The overview shows a read-only Persona preview; editing happens in a dialog.
    await user.click(screen.getByRole("button", { name: "编辑 Persona" }));
    const box = screen.getByPlaceholderText("Leave blank to use the built-in default persona");
    await user.type(box, "You are a code archaeologist.");
    // Local state until the user commits — no save while typing.
    expect(save).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "保存" }));

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    expect(saved.agents[0].instructions).toBe("You are a code archaeologist.");
  });

  it("edits the Default Persona template (doc.default_instructions) and commits on blur", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });

    render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Default Persona" }));
    const box = screen.getByPlaceholderText("Leave blank to use the built-in default persona");
    await user.type(box, "You are a research copilot.");
    expect(save).not.toHaveBeenCalled(); // local state until blur
    await user.tab();

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    expect(saved.default_instructions).toBe("You are a research copilot.");
  });

  it("seeds a new agent's Instructions from the Default Persona template", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });

    const seeded: AgentConfigDocument = { ...baseDoc, default_instructions: "House voice." };
    render(<SettingsView doc={seeded} onBack={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "New agent" }));

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    expect(saved.agents).toHaveLength(2);
    expect(saved.agents[1].instructions).toBe("House voice.");
  });

  it("deletes an agent (two-step confirm) and reassigns the default", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });

    // Two agents, with the *selected* (main) one being the deployment default.
    const twoAgents: AgentConfigDocument = {
      ...baseDoc,
      default_agent: "main",
      agents: [
        baseDoc.agents[0],
        { ...baseDoc.agents[0], id: "agent-2", name: "Support" },
      ],
    };
    render(<SettingsView doc={twoAgents} onBack={vi.fn()} />);

    // First click arms the confirm; nothing persisted yet.
    await user.click(screen.getByRole("button", { name: /删除/ }));
    expect(save).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "确认删除" }));

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    expect(saved.agents).toHaveLength(1);
    expect(saved.agents[0].id).toBe("agent-2");
    // The default followed the survivor.
    expect(saved.default_agent).toBe("agent-2");
  });

  it("disables delete for the only agent", async () => {
    render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);
    expect(screen.getByRole("button", { name: /删除/ })).toBeDisabled();
  });

  it("sets a non-default agent as the deployment default", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });

    // main is default; select the non-default "Support" agent, then promote it.
    const twoAgents: AgentConfigDocument = {
      ...baseDoc,
      default_agent: "main",
      agents: [
        baseDoc.agents[0],
        { ...baseDoc.agents[0], id: "agent-2", name: "Support" },
      ],
    };
    render(<SettingsView doc={twoAgents} onBack={vi.fn()} />);

    await user.selectOptions(screen.getByLabelText("Select agent"), "agent-2");
    await user.click(screen.getByRole("button", { name: "设为默认" }));

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    expect(saved.default_agent).toBe("agent-2");
  });

  it("shows the inherited default and switches an agent to an override model", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });

    const docWithModels: AgentConfigDocument = {
      ...baseDoc,
      models: {
        default_model: "openai/gpt-4o-mini",
        providers: [
          { name: "dashscope", models: [{ id: "qwen-max", type: "chat" }] },
        ],
      },
      providers: [
        ...baseDoc.providers,
        {
          id: "llm.default",
          type: "llm",
          name: "Default model provider",
          status: "healthy",
          settings: { default_model: "dashscope/qwen-plus" },
          secret_configured: true,
          used_by: ["model"],
        },
      ],
    };

    render(<SettingsView doc={docWithModels} onBack={vi.fn()} />);

    // Basic info (name + model) is edited inline — no dialog.
    // Blank agent model → chip reports it inherits, and the inherit option names
    // the resolved runtime default (from the llm.default provider, not the catalog).
    expect(screen.getByText("Using system default")).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: "Inherit (uses dashscope/qwen-plus)" })
    ).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Model"), "dashscope/qwen-max");

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    expect(saved.agents[0].model).toBe("dashscope/qwen-max");
  });

  it("flags Control Room tabs by provisioning readiness", () => {
    // baseDoc has no healthy llm.default provider → Connections needs setup;
    // the local vector engine and unbroken core tools read as ready.
    render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);

    expect(screen.getByTitle("Connections — Needs setup")).toBeInTheDocument();
    expect(screen.getByTitle("Knowledge Base — Ready")).toBeInTheDocument();

    // Once a healthy default model provider exists, Connections reads as ready.
    const ready: AgentConfigDocument = {
      ...baseDoc,
      providers: [
        ...baseDoc.providers,
        {
          id: "llm.default",
          type: "llm",
          name: "Default model provider",
          status: "healthy",
          settings: { default_model: "dashscope/qwen-max" },
          secret_configured: true,
          used_by: ["model"],
        },
      ],
    };
    render(<SettingsView doc={ready} onBack={vi.fn()} />);
    expect(screen.getAllByTitle("Connections — Ready").length).toBeGreaterThan(0);
  });

  it("scopes an agent to a subset of knowledge bases", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });
    listKnowledgeBases.mockResolvedValue([
      { id: "kb_a", name: "Handbook", visibility: "workspace" },
      { id: "kb_b", name: "Secrets", visibility: "private" },
    ]);

    render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);

    // Knowledge scoping lives in a dialog opened from the overview preview card.
    await user.click(screen.getByRole("button", { name: "编辑知识库" }));
    // Loaded async → the KB appears once the promise resolves.
    const handbook = await screen.findByText("Handbook");
    await user.click(handbook);

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    expect(saved.agents[0].knowledge.kb_ids).toEqual(["kb_a"]);
  });
});
