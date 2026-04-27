import { proxyToBackend } from "@/lib/backend";

export async function GET(_: Request, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  return proxyToBackend(`/v1/sessions/${sessionId}`);
}

export async function DELETE(_: Request, { params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  return proxyToBackend(`/v1/sessions/${sessionId}`, { method: "DELETE" });
}
