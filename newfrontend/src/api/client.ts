import { parseSSE } from "../lib/sse";

export interface ResponseStreamParams {
  model: string;
  input: string;
  user_id: string;
  conversation?: string;
  previous_response_id?: string;
  background?: boolean;
}

export function streamResponse(
  params: ResponseStreamParams,
  signal: AbortSignal
): AsyncIterable<unknown> {
  return (async function* () {
    const res = await fetch("/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...params, store: true, stream: true }),
      signal,
    });
    if (!res.ok || !res.body) throw new Error(`stream failed: ${res.status}`);
    yield* parseSSE(res.body);
  })();
}
