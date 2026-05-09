import { proxyToBackend } from "@/lib/backend";

export async function POST(request: Request, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  return proxyToBackend(`/v1/sessions/${sessionId}/cancel`, { method: "POST" });
}
