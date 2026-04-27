import type { AuthResponse, SessionDetail, SessionSummary, StreamEvent, UserProfile } from "@/lib/types";

const TOKEN_KEY = "pai-rag.auth_token";

function assertOk(response: Response) {
  if (response.ok) {
    return Promise.resolve();
  }
  return response.text().then((text) => {
    throw new Error(`${response.status}: ${text.slice(0, 600)}`);
  });
}

export function getAuthToken() {
  if (typeof window === "undefined") {
    return "";
  }
  return window.localStorage.getItem(TOKEN_KEY) || "";
}

export function setAuthToken(token: string) {
  if (typeof window !== "undefined") {
    window.localStorage.setItem(TOKEN_KEY, token);
  }
}

export function clearAuthToken() {
  if (typeof window !== "undefined") {
    window.localStorage.removeItem(TOKEN_KEY);
  }
}

function authHeaders(extra?: HeadersInit) {
  const headers = new Headers(extra);
  const token = getAuthToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return headers;
}

export async function login(username: string, password: string): Promise<AuthResponse> {
  const response = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  await assertOk(response);
  return response.json();
}

export async function register(username: string, password: string): Promise<AuthResponse> {
  const response = await fetch("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  await assertOk(response);
  return response.json();
}

export async function currentUser(): Promise<UserProfile> {
  const response = await fetch("/api/auth/me", {
    headers: authHeaders(),
    cache: "no-store",
  });
  await assertOk(response);
  return response.json();
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await fetch("/api/sessions", { headers: authHeaders(), cache: "no-store" });
  await assertOk(response);
  const payload = await response.json();
  return payload.data ?? [];
}

export async function createSession(): Promise<SessionDetail> {
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({}),
  });
  await assertOk(response);
  return response.json();
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  const response = await fetch(`/api/sessions/${sessionId}`, { headers: authHeaders(), cache: "no-store" });
  await assertOk(response);
  return response.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}`, { method: "DELETE", headers: authHeaders() });
  await assertOk(response);
}

export async function cancelSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}/cancel`, { method: "POST", headers: authHeaders() });
  await assertOk(response);
}

export async function streamAgentPrompt(
  sessionId: string,
  text: string,
  onChunk: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch(`/api/agent/sessions/${sessionId}/prompt`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ message: text }),
    signal,
  });
  await assertOk(response);

  const returnedSessionId = response.headers.get("X-Session-Id") || sessionId;
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
        const payload = JSON.parse(data);
        if (payload.update) {
          onChunk({ sessionId: payload.sessionId || returnedSessionId, update: payload.update });
        }
      }
    }
  }

  return returnedSessionId;
}
