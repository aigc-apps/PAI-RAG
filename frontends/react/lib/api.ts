import type { ChatMessage, SessionDetail, SessionSummary, StreamEvent } from "@/lib/types";

function assertOk(response: Response) {
  if (response.ok) {
    return Promise.resolve();
  }
  return response.text().then((text) => {
    throw new Error(`${response.status}: ${text.slice(0, 600)}`);
  });
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await fetch("/api/sessions", { cache: "no-store" });
  await assertOk(response);
  const payload = await response.json();
  return payload.data ?? [];
}

export async function createSession(): Promise<SessionDetail> {
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  await assertOk(response);
  return response.json();
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  const response = await fetch(`/api/sessions/${sessionId}`, { cache: "no-store" });
  await assertOk(response);
  return response.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
  await assertOk(response);
}

export async function cancelSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}/cancel`, { method: "POST" });
  await assertOk(response);
}

export async function streamChat(
  sessionId: string | null,
  text: string,
  onChunk: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch("/api/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(sessionId ? { "X-Session-Id": sessionId } : {}),
    },
    body: JSON.stringify({
      model: "hermes-agent",
      stream: true,
      messages: [{ role: "user", content: text } satisfies ChatMessage],
    }),
    signal,
  });
  await assertOk(response);

  const returnedSessionId = response.headers.get("X-Session-Id") || sessionId || "";
  const reader = response.body?.getReader();
  if (!reader) {
    return returnedSessionId;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const event of events) {
      const lines = event.split("\n").filter((line) => line.startsWith("data: "));
      for (const line of lines) {
        const data = line.slice(6);
        if (data === "[DONE]") {
          return returnedSessionId;
        }
        const payload = JSON.parse(data);
        for (const choice of payload.choices ?? []) {
          const content = choice.delta?.content;
          if (content) {
            onChunk({ sessionId: returnedSessionId, content });
          }
        }
      }
    }
  }

  return returnedSessionId;
}
