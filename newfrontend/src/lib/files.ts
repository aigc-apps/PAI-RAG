import { getUserId } from "./user";

/** Same-origin URL for a sandbox artifact, scoped to the current user. Vite
 * proxies `/v1` to the backend in dev. */
export function fileUrl(id: string): string {
  return `/v1/files/${encodeURIComponent(id)}?user_id=${encodeURIComponent(getUserId())}`;
}

export function humanSize(n: number): string {
  if (n < 0 || !Number.isFinite(n)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return i === 0 ? `${v} ${units[i]}` : `${v.toFixed(1)} ${units[i]}`;
}
