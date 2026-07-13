import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useI18nStore } from "../../i18n";
import type { KnowledgeBase } from "../../api/knowledge";

const listKnowledgeBases = vi.fn();
const listDataSources = vi.fn();
vi.mock("../../api/knowledge", async (importActual) => ({
  ...(await importActual<typeof import("../../api/knowledge")>()),
  listKnowledgeBases: () => listKnowledgeBases(),
  listDataSources: () => listDataSources(),
}));

vi.mock("../../api/models", () => ({
  listModels: vi.fn().mockResolvedValue({ ids: [], default: "", models: [], defaultEmbedding: null, defaultRerank: null }),
  modelsByType: vi.fn().mockReturnValue([]),
}));

import { KnowledgeView } from "../KnowledgeView";

const kb: KnowledgeBase = {
  id: "kb_1",
  name: "Production KB",
  description: "Production docs",
  owner_user_id: "u1",
  visibility: "private",
  status: "ready",
  document_count: 2,
  chunk_count: 4,
  default_parser_config: {},
  default_retrieval_config: {},
  embedding_config: {},
  vector_store_config: {},
  keyword_index_config: {},
  rerank_config: {},
  created_at: "2026-07-13T00:00:00Z",
  updated_at: "2026-07-13T00:00:00Z",
};

beforeEach(() => {
  useI18nStore.getState().setLang("zh");
  listKnowledgeBases.mockReset();
  listKnowledgeBases.mockResolvedValue([kb]);
  listDataSources.mockReset();
  listDataSources.mockResolvedValue([]);
});

describe("KnowledgeView routes", () => {
  it("opens a routed KB tab and reports tab navigation", async () => {
    useI18nStore.getState().setLang("en");
    const onTabChange = vi.fn();
    render(
      <KnowledgeView
        onBack={vi.fn()}
        kbId="kb_1"
        tab="datasources"
        onOpenKb={vi.fn()}
        onBackToList={vi.fn()}
        onTabChange={onTabChange}
        onInvalidKb={vi.fn()}
      />,
    );

    expect(await screen.findByText("Production KB")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /documents/i }));
    expect(onTabChange).toHaveBeenCalledWith("files");
  });

  it("reports an unknown routed KB after loading", async () => {
    const onInvalidKb = vi.fn();
    render(
      <KnowledgeView
        onBack={vi.fn()}
        kbId="missing"
        tab="overview"
        onOpenKb={vi.fn()}
        onBackToList={vi.fn()}
        onTabChange={vi.fn()}
        onInvalidKb={onInvalidKb}
      />,
    );

    await waitFor(() => expect(onInvalidKb).toHaveBeenCalledOnce());
  });
});
