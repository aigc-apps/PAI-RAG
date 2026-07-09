import { describe, it, expect, vi, beforeEach } from "vitest";
import { streamResponse } from "../client";

function streamFrom(parts: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({ start(c) { for (const p of parts) c.enqueue(enc.encode(p)); c.close(); } });
}

describe("streamResponse", () => {
  beforeEach(() => vi.restoreAllMocks());
  it("POSTs /v1/responses with stream+background and yields parsed events", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200,
      body: streamFrom(['data: {"type":"response.created","response":{"id":"r1"}}\n\n',
                        'data: {"type":"response.completed","response":{"id":"r1","status":"completed"}}\n\n']),
    } as unknown as Response);
    vi.stubGlobal("fetch", fetchMock);
    const out: any[] = [];
    for await (const e of streamResponse(
      { model: "m", input: "hi", background: true },
      new AbortController().signal)) out.push(e);
    expect(out.map((e) => e.type)).toEqual(["response.created", "response.completed"]);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/v1/responses");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body);
    expect(body.stream).toBe(true);
    expect(body.store).toBe(true);
    expect(body.background).toBe(true);
  });
});
