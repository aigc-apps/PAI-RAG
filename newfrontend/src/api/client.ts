import OpenAI from "openai";
import type { ResponseStreamEvent } from "openai/resources/responses/responses";

// The real API key lives server-side in the lean service; the browser only ever
// talks to our own service via the Vite dev proxy, so a placeholder + browser
// usage is acceptable for v1 (no auth layer yet).
const client = new OpenAI({
  baseURL: `${window.location.origin}/v1`,
  apiKey: "sk-noauth",
  dangerouslyAllowBrowser: true,
});

export interface ResponseStreamParams {
  model: string;
  input: string;
  user_id: string;
  conversation?: string;
  previous_response_id?: string;
}

export function streamResponse(
  params: ResponseStreamParams,
  signal: AbortSignal
): AsyncIterable<ResponseStreamEvent> {
  // store=true so the backend persists the turn + conversation row.
  // `as never` on params bypasses the SDK's strict union type for our extra user_id field;
  // `as unknown as AsyncIterable<ResponseStreamEvent>` recovers the correct iterable type
  // that `as never` loses (the SDK's Stream<T> implements AsyncIterable<T> at runtime).
  return client.responses.create(
    { ...params, store: true, stream: true } as never,
    { signal }
  ) as unknown as AsyncIterable<ResponseStreamEvent>;
}
