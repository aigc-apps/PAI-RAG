import { describe, it, expect, vi, beforeEach } from "vitest";
import { listModels } from "../models";

describe("listModels", () => {
  beforeEach(() => vi.restoreAllMocks());
  it("returns the model ids, types and defaults from /v1/models", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({
        object: "list", default: "openai/fast",
        default_embedding: "ds/emb", default_rerank: "ds/rr",
        data: [
          { id: "fast", type: "chat" },
          { id: "ds/emb", type: "embedding", dimension: 1024 },
          { id: "ds/rr", type: "rerank" },
        ],
      }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);
    expect(await listModels()).toEqual({
      ids: ["fast", "ds/emb", "ds/rr"],
      default: "openai/fast",
      defaultEmbedding: "ds/emb",
      defaultRerank: "ds/rr",
      models: [
        { id: "fast", type: "chat", dimension: null },
        { id: "ds/emb", type: "embedding", dimension: 1024 },
        { id: "ds/rr", type: "rerank", dimension: null },
      ],
    });
    expect(String(fetchMock.mock.calls[0][0])).toContain("/v1/models");
  });
  it("falls back to first id when default is absent", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ object: "list", data: [{ id: "fast" }] }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);
    expect((await listModels()).default).toBe("fast");
  });
  it("throws on a non-ok response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 } as Response));
    await expect(listModels()).rejects.toThrow();
  });
});
