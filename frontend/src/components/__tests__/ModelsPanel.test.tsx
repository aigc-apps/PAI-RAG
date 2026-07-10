import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ModelsPanel } from "../ModelsPanel";
import { useAgentConfigStore } from "../../store/agentConfig";
import type { AgentConfigDocument, ModelCatalogDoc } from "../../api/agentConfig";

function docWith(models: ModelCatalogDoc): AgentConfigDocument {
  return {
    setup: { completed: true, skipped_steps: [] },
    models,
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
    providers: [],
    capabilities: [],
  };
}

const withProvider = docWith({
  providers: [
    { name: "dashscope", base_url: "https://ds/v1", api_key_env: "DASHSCOPE_API_KEY", models: [] },
  ],
});

let save: ReturnType<typeof vi.fn>;

beforeEach(() => {
  save = vi.fn(async (doc: AgentConfigDocument) => doc);
  useAgentConfigStore.setState({ loading: false, error: undefined, save });
});

describe("ModelsPanel", () => {
  it("adds a model provider", async () => {
    const user = userEvent.setup();
    render(<ModelsPanel doc={docWith({})} />);

    await user.type(screen.getByLabelText("Provider name"), "dashscope");
    await user.type(
      screen.getByLabelText("Provider base URL"),
      "https://dashscope.aliyuncs.com/compatible-mode/v1"
    );
    await user.type(screen.getByLabelText("Provider API key env"), "DASHSCOPE_API_KEY");
    await user.click(screen.getByRole("button", { name: "Add" }));

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    const providers = saved.models.providers ?? [];
    expect(providers).toHaveLength(1);
    expect(providers[0]).toMatchObject({
      name: "dashscope",
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
      api_key_env: "DASHSCOPE_API_KEY",
      models: [],
    });
  });

  it("registers an embedding model as the default for its type", async () => {
    const user = userEvent.setup();
    render(<ModelsPanel doc={withProvider} />);

    await user.selectOptions(screen.getByLabelText("Model provider"), "dashscope");
    await user.selectOptions(screen.getByLabelText("Model type"), "embedding");
    await user.type(screen.getByLabelText("Model id"), "text-embedding-v4");
    await user.selectOptions(screen.getByLabelText("Model protocol"), "dashscope");
    await user.type(screen.getByLabelText("Embedding dimension"), "1024");
    // "Set as default" is checked by default.
    await user.click(screen.getByRole("button", { name: "Register" }));

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    const provider = (saved.models.providers ?? []).find((p) => p.name === "dashscope");
    expect(provider?.models).toEqual([
      { id: "text-embedding-v4", type: "embedding", protocol: "dashscope", dimension: 1024 },
    ]);
    expect(saved.models.default_embedding_model).toBe("dashscope/text-embedding-v4");
    // The provider's shared connection is untouched.
    expect(provider?.base_url).toBe("https://ds/v1");
  });

  it("registers a rerank model and hides the dimension field", async () => {
    const user = userEvent.setup();
    render(<ModelsPanel doc={withProvider} />);

    await user.selectOptions(screen.getByLabelText("Model provider"), "dashscope");
    await user.selectOptions(screen.getByLabelText("Model type"), "rerank");
    // Dimension only applies to embeddings.
    expect(screen.queryByLabelText("Embedding dimension")).not.toBeInTheDocument();
    await user.type(screen.getByLabelText("Model id"), "gte-rerank");
    await user.click(screen.getByRole("button", { name: "Register" }));

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    const provider = (saved.models.providers ?? []).find((p) => p.name === "dashscope");
    expect(provider?.models?.[0]).toMatchObject({ id: "gte-rerank", type: "rerank" });
    expect(provider?.models?.[0].dimension).toBeUndefined();
    expect(saved.models.default_rerank_model).toBe("dashscope/gte-rerank");
  });

  it("deleting a provider clears a default that pointed at its model", async () => {
    const user = userEvent.setup();
    const doc = docWith({
      default_embedding_model: "dashscope/text-embedding-v4",
      providers: [
        {
          name: "dashscope",
          base_url: "https://ds/v1",
          api_key_env: "DASHSCOPE_API_KEY",
          models: [{ id: "text-embedding-v4", type: "embedding", dimension: 1024 }],
        },
      ],
    });
    render(<ModelsPanel doc={doc} />);

    await user.click(screen.getByRole("button", { name: "Delete provider dashscope" }));

    expect(save).toHaveBeenCalledOnce();
    const saved = save.mock.calls[0][0] as AgentConfigDocument;
    expect(saved.models.providers).toHaveLength(0);
    expect(saved.models.default_embedding_model).toBeUndefined();
  });
});
