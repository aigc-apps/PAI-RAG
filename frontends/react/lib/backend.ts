const DEFAULT_BACKEND = "http://127.0.0.1:8000";

export function backendBaseUrl() {
  return (process.env.BACKEND_BASE_URL || DEFAULT_BACKEND).replace(/\/$/, "");
}

export function backendHeaders(extra?: HeadersInit) {
  const headers = new Headers(extra);
  const apiKey = process.env.SERVER_API_KEY;
  if (apiKey) {
    headers.set("Authorization", `Bearer ${apiKey}`);
  }
  return headers;
}

export async function proxyToBackend(path: string, init: RequestInit = {}) {
  const response = await fetch(`${backendBaseUrl()}${path}`, {
    ...init,
    headers: backendHeaders(init.headers),
    cache: "no-store",
  });

  const headers = new Headers();
  const contentType = response.headers.get("content-type");
  const sessionId = response.headers.get("X-Session-Id");
  if (contentType) {
    headers.set("content-type", contentType);
  }
  if (sessionId) {
    headers.set("X-Session-Id", sessionId);
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
