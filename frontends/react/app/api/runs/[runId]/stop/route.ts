import { proxyToBackend } from "@/lib/backend";

export async function POST(request: Request, { params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  return proxyToBackend(
    `/v1/runs/${runId}/stop`,
    {
      method: "POST",
    },
  );
}
