import { describe, it, expect, vi, beforeEach } from "vitest";
import { cancelResponse, streamResume } from "../responses";

function streamFrom(parts: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      for (const p of parts) c.enqueue(enc.encode(p));
      c.close();
    },
  });
}

describe("responses api", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("cancelResponse POSTs to the cancel path and tolerates 404", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 404 } as Response);
    vi.stubGlobal("fetch", fetchMock);
    await cancelResponse("resp_1");
    expect(String(fetchMock.mock.calls[0][0])).toContain("/v1/responses/resp_1/cancel");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "POST" });
  });

  it("cancelResponse throws on a real failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 } as Response));
    await expect(cancelResponse("resp_1")).rejects.toThrow();
  });

  it("streamResume requests the cursor URL and yields parsed events", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      body: streamFrom(['data: {"type":"response.completed"}\n\n']),
    } as unknown as Response);
    vi.stubGlobal("fetch", fetchMock);
    const out: unknown[] = [];
    for await (const e of streamResume("resp_1", 7, new AbortController().signal)) out.push(e);
    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("/v1/responses/resp_1");
    expect(url).toContain("stream=true");
    expect(url).toContain("starting_after=7");
    expect(out).toEqual([{ type: "response.completed" }]);
  });

  it("streamResume throws when the run is not resumable (409)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 409 } as Response));
    const it = streamResume("resp_x", 0, new AbortController().signal);
    await expect((async () => { for await (const _ of it) { /* drain */ } })()).rejects.toThrow();
  });
});
