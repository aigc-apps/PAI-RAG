import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useI18nStore } from "../../i18n";
import type { KnowledgeBase } from "../../api/knowledge";

const listKnowledgeBases = vi.fn();
const listDataSources = vi.fn();
const listKnowledgeDocuments = vi.fn();
const searchKnowledge = vi.fn();
const updateKnowledgeBase = vi.fn();
vi.mock("../../api/knowledge", async (importActual) => ({
  ...(await importActual<typeof import("../../api/knowledge")>()),
  listKnowledgeBases: () => listKnowledgeBases(),
  listDataSources: () => listDataSources(),
  listKnowledgeDocuments: (...args: unknown[]) => listKnowledgeDocuments(...args),
  searchKnowledge: (...args: unknown[]) => searchKnowledge(...args),
  updateKnowledgeBase: (...args: unknown[]) => updateKnowledgeBase(...args),
}));

vi.mock("../../api/models", () => ({
  listModels: vi.fn().mockResolvedValue({ ids: [], default: "", models: [], defaultEmbedding: null, defaultRerank: null }),
  modelsByType: vi.fn().mockReturnValue([]),
}));

import { KnowledgeView, SyncProgressView } from "../KnowledgeView";

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
  listKnowledgeDocuments.mockReset();
  listKnowledgeDocuments.mockResolvedValue({
    data: [], total: 0, offset: 0, limit: 50, has_more: false,
  });
  searchKnowledge.mockReset();
  updateKnowledgeBase.mockReset();
  updateKnowledgeBase.mockResolvedValue(kb);
});

describe("KnowledgeView routes", () => {
  it("keeps the embedded documents workspace readable at wide settings widths", async () => {
    listKnowledgeDocuments.mockResolvedValue({
      data: [{
        id: "doc_1",
        kb_id: "kb_1",
        uri: "https://docs.example.com/products/turbox/production/deployment-guide",
        source_type: "website",
        title: "PAI-TurboX：面向生产环境的完整部署与故障排查指南",
        description: "Production deployment guide",
        status: "indexed",
        tags: ["PAI-TurboX", "production"],
        chunk_count: 54,
        indexed_at: "2026-07-14T00:00:00Z",
      }],
      total: 1, offset: 0, limit: 50, has_more: false,
    });

    render(
      <KnowledgeView
        onBack={vi.fn()}
        kbId="kb_1"
        tab="files"
        embedded
        onOpenKb={vi.fn()}
        onBackToList={vi.fn()}
        onTabChange={vi.fn()}
        onInvalidKb={vi.fn()}
      />,
    );

    expect(await screen.findByTestId("knowledge-documents-table")).toHaveClass(
      "min-w-[940px]",
      "table-fixed",
    );
    expect(screen.getByTestId("knowledge-embedded-detail")).toHaveClass("min-w-0");
    expect(screen.getByTestId("knowledge-tabbar")).toHaveClass("overflow-x-auto");
    expect(screen.getByTestId("knowledge-document-title-cell")).toHaveClass(
      "min-w-[300px]",
    );
  });

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

  it("shows only the final recall score and no KB rerank controls", async () => {
    useI18nStore.getState().setLang("en");
    listKnowledgeBases.mockResolvedValue([
      { ...kb, default_retrieval_config: { mode: "vector" } },
    ]);
    searchKnowledge.mockResolvedValue({
      data: [{
        kb_id: "kb_1", document_id: "doc_1", chunk_id: "chunk_1",
        title: "Production guide", source_uri: "https://docs.example/guide",
        source_type: "website", text: "TurboX production setup",
        score: 0.876, vector_score: 0.8, keyword_score: 0.4, metadata: {},
      }],
      total: 1, offset: 0, limit: 6, has_more: false,
    });

    const props = {
      onBack: vi.fn(), kbId: "kb_1", onOpenKb: vi.fn(),
      onBackToList: vi.fn(), onTabChange: vi.fn(), onInvalidKb: vi.fn(),
    };
    const { rerender } = render(<KnowledgeView {...props} tab="recall" />);
    await screen.findByText("Production KB");
    await userEvent.type(screen.getByPlaceholderText(/enter a query/i), "turbox");
    await userEvent.click(screen.getByRole("button", { name: /^search$/i }));

    expect(await screen.findByText("Final score")).toBeInTheDocument();
    expect(screen.getByText("0.876")).toBeInTheDocument();
    expect(screen.queryByText("0.80")).not.toBeInTheDocument();
    expect(screen.queryByText("0.40")).not.toBeInTheDocument();
    expect(screen.queryByText(/^mode$/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "vector" })).not.toBeInTheDocument();
    expect(searchKnowledge).toHaveBeenCalledWith(
      expect.objectContaining({ mode: "hybrid" }),
    );
    expect(screen.getByText(/showing first 1/i)).toBeInTheDocument();
    expect(screen.queryByText(/mode vector/i)).not.toBeInTheDocument();

    rerender(<KnowledgeView {...props} tab="config" />);
    await waitFor(() => expect(screen.queryByText("Reranker")).not.toBeInTheDocument());
  });

  it("normalizes stored retrieval mode to hybrid when saving config", async () => {
    useI18nStore.getState().setLang("en");
    listKnowledgeBases.mockResolvedValue([
      {
        ...kb,
        default_retrieval_config: {
          mode: "vector",
          top_k: 6,
          score_threshold: 0,
          force_citation: true,
        },
      },
    ]);
    render(
      <KnowledgeView
        onBack={vi.fn()}
        kbId="kb_1"
        tab="config"
        onOpenKb={vi.fn()}
        onBackToList={vi.fn()}
        onTabChange={vi.fn()}
        onInvalidKb={vi.fn()}
      />,
    );

    const topK = await screen.findByLabelText("top_k");
    await userEvent.clear(topK);
    await userEvent.type(topK, "8");
    expect(screen.queryByText(/^mode$/i)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /save configuration/i }));

    await waitFor(() =>
      expect(updateKnowledgeBase).toHaveBeenCalledWith(
        "kb_1",
        expect.objectContaining({
          default_retrieval_config: expect.objectContaining({ mode: "hybrid" }),
        }),
      ),
    );
  });
});

describe("SyncProgressView", () => {
  it("renders accessible progress, rate, ETA, failures, and cancellation", async () => {
    const onCancel = vi.fn();
    render(
      <SyncProgressView
        cancelling={false}
        onCancel={onCancel}
        progress={{
          phase: "indexing",
          total: 2144,
          discovered: 2144,
          fetched: 200,
          embedded: 175,
          persisted: 150,
          indexed: 125,
          unchanged: 0,
          deleted: 0,
          failed: 1,
          bytes_fetched: 1000,
          docs_per_second: 4.8,
          estimated_seconds_remaining: 415,
        }}
      />
    );

    expect(screen.getByRole("progressbar", { name: "同步进度" })).toHaveAttribute(
      "aria-valuenow",
      "6"
    );
    expect(screen.getByText("126 / 2144")).toBeInTheDocument();
    expect(screen.getByText("4.8 篇/秒")).toBeInTheDocument();
    expect(screen.getByText("预计剩余 415 秒")).toBeInTheDocument();
    expect(screen.getByText("1 篇失败")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "取消同步" }));
    expect(onCancel).toHaveBeenCalledOnce();
  });
});
