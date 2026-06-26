import type { ConversationDetail, ConversationSummary } from "../types";

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function listConversations(
  userId: string
): Promise<ConversationSummary[]> {
  const res = await fetch(
    `/v1/conversations?user_id=${encodeURIComponent(userId)}`
  );
  const body = await jsonOrThrow<{ data: ConversationSummary[] }>(res);
  return body.data;
}

export async function getConversation(
  id: string
): Promise<ConversationDetail> {
  const res = await fetch(`/v1/conversations/${encodeURIComponent(id)}`);
  return jsonOrThrow<ConversationDetail>(res);
}

export async function deleteConversation(id: string): Promise<void> {
  const res = await fetch(`/v1/conversations/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`delete failed: ${res.status}`);
  }
}
