const DEFAULT_BACKEND = "http://127.0.0.1:8000";

export function backendBaseUrl() {
  return (process.env.BACKEND_BASE_URL || DEFAULT_BACKEND).replace(/\/$/, "");
}

export function backendHeaders(extra?: HeadersInit) {
  return new Headers(extra);
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
  const runId = response.headers.get("X-Run-Id");
  const cacheControl = response.headers.get("cache-control");
  const buffering = response.headers.get("x-accel-buffering");
  if (contentType) {
    headers.set("content-type", contentType);
  }
  if (sessionId) {
    headers.set("X-Session-Id", sessionId);
  }
  if (runId) {
    headers.set("X-Run-Id", runId);
  }
  if (cacheControl) {
    headers.set("cache-control", cacheControl);
  }
  if (buffering) {
    headers.set("x-accel-buffering", buffering);
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
