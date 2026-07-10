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

const baseDoc: AgentConfigDocument = {
  setup: { completed: true, skipped_steps: [] },
  models: {},
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

// A doc with the sandbox capability turned on, so the code-manifest section renders.
const sandboxOnDoc: AgentConfigDocument = {
  ...baseDoc,
  capabilities: baseDoc.capabilities.map((c) =>
    c.id === "sandbox" ? { ...c, enabled: true, permission: "auto", status: "ready" } : c
  ),
};

describe("SettingsView", () => {
  beforeEach(() => {
    generateCodeManifest.mockReset();
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

  it("hides the code manifest section when the sandbox capability is off", () => {
    // baseDoc has sandbox capability disabled; the Agents tab is the default.
    render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);
    expect(screen.queryByText("代码库配置单")).not.toBeInTheDocument();
  });

  it("generates a code manifest and saves it onto the agent", async () => {
    const user = userEvent.setup();
    const save = vi.fn(async (doc: AgentConfigDocument) => doc);
    useAgentConfigStore.setState({ save });
    generateCodeManifest.mockResolvedValue({ manifest: "- repo-a — the API server" });

    render(<SettingsView doc={sandboxOnDoc} onBack={vi.fn()} />);

    // Section is visible once sandbox is enabled.
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
});
