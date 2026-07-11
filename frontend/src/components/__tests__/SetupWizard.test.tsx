import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SetupWizard } from "../SetupWizard";
import { useAgentConfigStore } from "../../store/agentConfig";
import type { AgentConfigDocument, ProviderStatus } from "../../api/agentConfig";

function docWithLlm(status: ProviderStatus, defaultModel = "dashscope/qwen-max"): AgentConfigDocument {
  return {
    setup: { completed: false, skipped_steps: [] },
    models: { default_model: defaultModel, providers: [] },
    soul: { name: "", role: "", identity: "", personality: [], principles: [], expertise: [], style: "", constraints: [] },
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
    skills: { root: "./data/skills" },
    default_agent: "main",
    agents: [],
    providers: [
      {
        id: "llm.default",
        type: "llm",
        name: "Default model provider",
        status,
        settings: { default_model: defaultModel },
        secret_configured: status === "healthy",
        used_by: ["model"],
      },
    ],
    capabilities: [],
  };
}

let completeSetup: ReturnType<typeof vi.fn>;

beforeEach(() => {
  completeSetup = vi.fn(async (setup) => ({ ...docWithLlm("healthy"), setup: { ...setup } }));
  useAgentConfigStore.setState({ loading: false, error: undefined, completeSetup });
});

describe("SetupWizard", () => {
  it("blocks Finish until a model connects", () => {
    render(<SetupWizard doc={docWithLlm("missing_config")} onDone={vi.fn()} />);

    // The required step is flagged and Finish is disabled.
    expect(screen.getByText("Required")).toBeInTheDocument();
    const finish = screen.getByRole("button", { name: /Finish setup/ });
    expect(finish).toBeDisabled();
  });

  it("enables Finish once the model is healthy and completes setup", async () => {
    const user = userEvent.setup();
    const onDone = vi.fn();
    render(<SetupWizard doc={docWithLlm("healthy")} onDone={onDone} />);

    expect(screen.getByText("Ready")).toBeInTheDocument();
    const finish = screen.getByRole("button", { name: /Finish setup/ });
    expect(finish).toBeEnabled();

    await user.click(finish);
    expect(completeSetup).toHaveBeenCalledOnce();
    expect(completeSetup.mock.calls[0][0]).toMatchObject({ completed: true });
  });
});
