import { parseSSE } from "../lib/sse";

export async function cancelResponse(id: string): Promise<void> {
  const res = await fetch(`/v1/responses/${encodeURIComponent(id)}/cancel`, {
    method: "POST",
  });
  // 404 == the run already finished/unknown server-side — treat as a no-op.
  if (!res.ok && res.status !== 404) {
    throw new Error(`cancel failed: ${res.status}`);
  }
}

export function streamResume(
  id: string,
  startingAfter: number,
  signal: AbortSignal
): AsyncIterable<unknown> {
  return (async function* () {
    const res = await fetch(
      `/v1/responses/${encodeURIComponent(id)}?stream=true&starting_after=${startingAfter}`,
      { signal }
    );
    if (!res.ok || !res.body) {
      throw new Error(`resume failed: ${res.status}`);
    }
    yield* parseSSE(res.body);
  })();
}
