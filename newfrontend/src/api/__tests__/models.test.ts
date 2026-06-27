import { describe, it, expect, vi, beforeEach } from "vitest";
import { listModels } from "../models";

describe("listModels", () => {
  beforeEach(() => vi.restoreAllMocks());
  it("returns the model ids from /v1/models", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ object: "list", data: [{ id: "fast" }, { id: "smart" }] }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);
    expect(await listModels()).toEqual(["fast", "smart"]);
    expect(String(fetchMock.mock.calls[0][0])).toContain("/v1/models");
  });
  it("throws on a non-ok response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 } as Response));
    await expect(listModels()).rejects.toThrow();
  });
});
