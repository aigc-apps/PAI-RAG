import { proxyToBackend } from "@/lib/backend";

export async function POST(request: Request) {
  const sessionId = request.headers.get("X-Session-Id");
  return proxyToBackend(
    "/v1/runs",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(sessionId ? { "X-Session-Id": sessionId } : {}),
      },
      body: await request.text(),
    },
    request,
  );
}
