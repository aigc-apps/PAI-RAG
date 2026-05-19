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
  const cacheControl = response.headers.get("cache-control");
  const buffering = response.headers.get("x-accel-buffering");
  if (contentType) {
    headers.set("content-type", contentType);
  }
  if (cacheControl) {
    headers.set("cache-control", cacheControl);
  }
  if (buffering) {
    headers.set("x-accel-buffering", buffering);
  }

  // For streaming responses (SSE), passing `response.body` directly to
  // `new Response()` lets Next.js's response pipeline buffer until the
  // upstream completes. Re-pump through an explicit ReadableStream that
  // enqueues every chunk as it arrives — Next then has no opportunity to
  // batch, and the first byte reaches the browser within milliseconds.
  const isStream = (contentType || "").includes("text/event-stream");
  if (!isStream || !response.body) {
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  }

  const upstream = response.body.getReader();
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { value, done } = await upstream.read();
        if (done) {
          controller.close();
          return;
        }
        controller.enqueue(value);
      } catch (err) {
        controller.error(err);
      }
    },
    cancel(reason) {
      upstream.cancel(reason).catch(() => {});
    },
  });

  return new Response(stream, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
