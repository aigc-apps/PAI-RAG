// A thin wrapper around `fetch` for the app's same-origin API.
//
// The browser never touches the JWT: it rides in an httpOnly cookie that the
// backend sets at login, and the Vite dev proxy forwards `/v1` same-origin. We
// only need to (a) always send that cookie and (b) notice when the server says
// the session is gone (401) so the app can drop back to the login screen.

type UnauthorizedHandler = () => void;

let onUnauthorized: UnauthorizedHandler | null = null;

/** Registered by the auth store so a 401 anywhere flips the app to logged-out. */
export function setUnauthorizedHandler(fn: UnauthorizedHandler | null): void {
  onUnauthorized = fn;
}

export async function apiFetch(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> {
  const res = await fetch(input, { credentials: "same-origin", ...init });
  if (res.status === 401) {
    // Session expired or was revoked (e.g. the account was disabled). Let the
    // store clear auth state; individual callers still see the 401 to handle.
    onUnauthorized?.();
  }
  return res;
}
