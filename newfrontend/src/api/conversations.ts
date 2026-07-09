import type { ConversationDetail, ConversationSummary } from "../types";
import { apiFetch } from "../lib/apiFetch";

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

// Identity is derived server-side from the session cookie; the client no longer
// passes a user_id (doing so could not impersonate anyone anyway).
export async function listConversations(): Promise<ConversationSummary[]> {
  const res = await apiFetch("/v1/conversations");
  const body = await jsonOrThrow<{ data: ConversationSummary[] }>(res);
  return body.data;
}

export async function getConversation(
  id: string
): Promise<ConversationDetail> {
  const res = await apiFetch(`/v1/conversations/${encodeURIComponent(id)}`);
  return jsonOrThrow<ConversationDetail>(res);
}

/** Drop the conversation's last turn so a regenerate re-runs the prompt in place
 * instead of appending. Returns the new anchor (the response before the removed
 * turn, or null if it was the first). */
export async function truncateLastTurn(
  conversationId: string,
  responseId: string
): Promise<{ previous_response_id: string | null }> {
  const res = await apiFetch(
    `/v1/conversations/${encodeURIComponent(conversationId)}/truncate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ response_id: responseId }),
    }
  );
  return jsonOrThrow<{ previous_response_id: string | null }>(res);
}

export async function deleteConversation(id: string): Promise<void> {
  const res = await apiFetch(`/v1/conversations/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`delete failed: ${res.status}`);
  }
}
