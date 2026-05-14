import { proxyToBackend } from "@/lib/backend";

export async function POST(request: Request) {
  const body = await request.text();
  return proxyToBackend("/v1/models/active", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
}
