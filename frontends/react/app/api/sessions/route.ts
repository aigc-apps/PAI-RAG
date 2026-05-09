import { proxyToBackend } from "@/lib/backend";

export async function GET(request: Request) {
  return proxyToBackend("/v1/sessions");
}

export async function POST(request: Request) {
  return proxyToBackend("/v1/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: await request.text(),
  });
}
