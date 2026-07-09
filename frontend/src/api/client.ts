import { parseSSE } from "../lib/sse";
import { apiFetch } from "../lib/apiFetch";

export interface ResponseStreamParams {
  model: string;
  agent_id?: string;
  input: string;
  conversation?: string;
  previous_response_id?: string;
  background?: boolean;
}

export function streamResponse(
  params: ResponseStreamParams,
  signal: AbortSignal
): AsyncIterable<unknown> {
  return (async function* () {
    const res = await apiFetch("/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...params, store: true, stream: true }),
      signal,
    });
    if (!res.ok || !res.body) {
      let message = `stream failed: ${res.status}`;
      try {
        const body = await res.json();
        if (body?.error?.message) message = body.error.message;
      } catch { /* response body not JSON */ }
      throw new Error(message);
    }
    yield* parseSSE(res.body);
  })();
}
