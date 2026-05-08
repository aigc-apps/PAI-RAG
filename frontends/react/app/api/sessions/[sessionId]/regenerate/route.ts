import { proxyToBackend } from "@/lib/backend";

export async function POST(request: Request, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  return proxyToBackend(
    `/v1/sessions/${sessionId}/regenerate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: await request.text(),
    },
    request,
  );
}
