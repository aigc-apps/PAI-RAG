import { describe, it, expect } from "vitest";
import { parseSSE } from "../sse";

function streamFrom(parts: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      for (const p of parts) c.enqueue(enc.encode(p));
      c.close();
    },
  });
}

async function collect(it: AsyncIterable<unknown>) {
  const out: unknown[] = [];
  for await (const x of it) out.push(x);
  return out;
}

describe("parseSSE", () => {
  it("yields each data event as parsed JSON", async () => {
    const out = await collect(
      parseSSE(streamFrom(['data: {"a":1}\n\n', 'data: {"b":2}\n\n']))
    );
    expect(out).toEqual([{ a: 1 }, { b: 2 }]);
  });

  it("reassembles events split across chunk boundaries", async () => {
    const out = await collect(parseSSE(streamFrom(['data: {"a":', '1}\n', "\n"])));
    expect(out).toEqual([{ a: 1 }]);
  });

  it("skips [DONE] sentinels", async () => {
    const out = await collect(
      parseSSE(streamFrom(['data: {"a":1}\n\n', "data: [DONE]\n\n"]))
    );
    expect(out).toEqual([{ a: 1 }]);
  });
});
