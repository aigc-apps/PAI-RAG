import { proxyToBackend } from "@/lib/backend";

export async function GET(request: Request) {
  return proxyToBackend("/v1/auth/me", {}, request);
}
