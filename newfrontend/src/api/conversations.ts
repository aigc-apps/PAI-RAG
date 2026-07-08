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

export async function deleteConversation(id: string): Promise<void> {
  const res = await apiFetch(`/v1/conversations/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`delete failed: ${res.status}`);
  }
}
