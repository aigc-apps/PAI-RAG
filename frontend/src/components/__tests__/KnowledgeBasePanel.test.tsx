import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { KnowledgeBasePanel } from "../KnowledgeBasePanel";
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

describe("KnowledgeBasePanel", () => {
  it("shows a read-only summary of embedding/rerank models", () => {
    const doc = docWith({
      default_embedding_model: "dashscope/text-embedding-v4",
      default_rerank_model: "dashscope/gte-rerank",
      providers: [
        {
          name: "dashscope",
          models: [
            { id: "text-embedding-v4", type: "embedding", dimension: 1024 },
            { id: "text-embedding-v3", type: "embedding", dimension: 1024 },
            { id: "gte-rerank", type: "rerank" },
          ],
        },
      ],
    });
    render(<KnowledgeBasePanel doc={doc} onConfigureVectorDB={vi.fn()} />);

    // Both defaults are surfaced, and every registered model of each role is
    // listed as a chip.
    expect(screen.getByText("Knowledge Base")).toBeInTheDocument();
    expect(screen.getAllByText("dashscope/text-embedding-v4").length).toBeGreaterThan(0);
    expect(screen.getByText("dashscope/text-embedding-v3")).toBeInTheDocument();
    expect(screen.getAllByText("dashscope/gte-rerank").length).toBeGreaterThan(0);
    // Vector DB is unconfigured (engine local by default) → shown as 未配置
    // (both the URL line and the status chip).
    expect(screen.getAllByText("未配置").length).toBeGreaterThan(0);
  });

  it("opens the vector database dialog via Configure", async () => {
    const user = userEvent.setup();
    const onConfigureVectorDB = vi.fn();
    render(<KnowledgeBasePanel doc={docWith({})} onConfigureVectorDB={onConfigureVectorDB} />);

    await user.click(screen.getByRole("button", { name: "Configure" }));
    expect(onConfigureVectorDB).toHaveBeenCalledOnce();
  });
});
