import { proxyToBackend } from "@/lib/backend";

export async function GET(request: Request, { params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  const url = new URL(request.url);
  const query = url.search || "";
  const lastEventId = request.headers.get("Last-Event-ID");
  return proxyToBackend(
    `/v1/runs/${runId}/events${query}`,
    {
      method: "GET",
      headers: lastEventId ? { "Last-Event-ID": lastEventId } : undefined,
    },
    request,
  );
}
