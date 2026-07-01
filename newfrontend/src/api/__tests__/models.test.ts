import { describe, it, expect, vi, beforeEach } from "vitest";
import { listModels } from "../models";

describe("listModels", () => {
  beforeEach(() => vi.restoreAllMocks());
  it("returns the model ids and default from /v1/models", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ object: "list", default: "openai/fast", data: [{ id: "fast" }, { id: "smart" }] }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);
    expect(await listModels()).toEqual({ ids: ["fast", "smart"], default: "openai/fast" });
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
