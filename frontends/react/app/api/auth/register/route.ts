import { proxyToBackend } from "@/lib/backend";

export async function POST(request: Request) {
  return proxyToBackend("/v1/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: await request.text(),
  }, request);
}
