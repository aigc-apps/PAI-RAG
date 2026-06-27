/** Parse a streaming SSE body into the JSON payloads of its `data:` lines,
 *  reassembling events that span read-chunk boundaries. */
export async function* parseSSE(
  stream: ReadableStream<Uint8Array>
): AsyncGenerator<unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload && payload !== "[DONE]") {
            yield JSON.parse(payload);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
