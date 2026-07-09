import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { apiFetch, setUnauthorizedHandler } from "../apiFetch";

function mockFetch(status: number) {
  return vi.fn().mockResolvedValue({ ok: status < 400, status } as Response);
}

describe("apiFetch", () => {
  beforeEach(() => vi.restoreAllMocks());
  afterEach(() => setUnauthorizedHandler(null));

  it("sends same-origin credentials", async () => {
    const f = mockFetch(200);
    vi.stubGlobal("fetch", f);
    await apiFetch("/v1/thing");
    expect(f.mock.calls[0][1]).toMatchObject({ credentials: "same-origin" });
  });

  it("invokes the unauthorized handler on a 401", async () => {
    vi.stubGlobal("fetch", mockFetch(401));
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    const res = await apiFetch("/v1/thing");
    expect(res.status).toBe(401);
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("does not invoke the handler on a successful response", async () => {
    vi.stubGlobal("fetch", mockFetch(200));
    const handler = vi.fn();
    setUnauthorizedHandler(handler);
    await apiFetch("/v1/thing");
    expect(handler).not.toHaveBeenCalled();
  });

  it("lets caller init override defaults but keeps returning the response", async () => {
    const f = mockFetch(500);
    vi.stubGlobal("fetch", f);
    const res = await apiFetch("/v1/thing", { method: "POST" });
    expect(res.status).toBe(500);
    expect(f.mock.calls[0][1]).toMatchObject({ method: "POST", credentials: "same-origin" });
  });
});
