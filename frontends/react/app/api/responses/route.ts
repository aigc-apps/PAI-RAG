import { proxyToBackend } from "@/lib/backend";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(request: Request) {
  const sessionId = request.headers.get("X-Session-Id");
  return proxyToBackend("/v1/responses", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...(sessionId ? { "X-Session-Id": sessionId } : {}),
    },
    body: await request.text(),
  });
}
