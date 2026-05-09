import { proxyToBackend } from "@/lib/backend";

export async function GET(request: Request, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  return proxyToBackend(`/v1/sessions/${sessionId}`);
}

export async function DELETE(request: Request, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  return proxyToBackend(`/v1/sessions/${sessionId}`, { method: "DELETE" });
}
