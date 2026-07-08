import { apiFetch } from "../lib/apiFetch";

export type Role = "admin" | "user";
export type UserStatus = "invited" | "active" | "disabled";

export interface AuthUser {
  id: string;
  email: string | null;
  role: Role;
  status: UserStatus;
  display_name: string | null;
}

interface AuthResponse {
  access_token: string;
  token_type: string;
  user: AuthUser;
}

export interface InviteResult {
  user: AuthUser;
  invite_token: string;
  invite_path: string;
  invite_url: string;
  expires_at: string;
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `request failed: ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) message = body.detail;
    } catch {
      /* not JSON */
    }
    throw new Error(message);
  }
  return (await res.json()) as T;
}

function jsonPost(path: string, body: unknown): Promise<Response> {
  return apiFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// --------------------------------------------------------------------------- //
// public
// --------------------------------------------------------------------------- //
export async function bootstrapStatus(): Promise<{ needed: boolean }> {
  return jsonOrThrow(await apiFetch("/v1/auth/bootstrap"));
}

export async function createAdmin(
  email: string,
  password: string,
  token?: string
): Promise<AuthUser> {
  const body = await jsonOrThrow<AuthResponse>(
    await jsonPost("/v1/auth/bootstrap", { email, password, token })
  );
  return body.user;
}

export async function login(email: string, password: string): Promise<AuthUser> {
  const body = await jsonOrThrow<AuthResponse>(
    await jsonPost("/v1/auth/login", { email, password })
  );
  return body.user;
}

export async function logout(): Promise<void> {
  await apiFetch("/v1/auth/logout", { method: "POST" });
}

export async function me(): Promise<AuthUser | null> {
  const res = await apiFetch("/v1/auth/me");
  if (res.status === 401) return null;
  const body = await jsonOrThrow<{ user: AuthUser }>(res);
  return body.user;
}

export async function acceptInvite(
  token: string,
  password: string
): Promise<AuthUser> {
  const body = await jsonOrThrow<AuthResponse>(
    await jsonPost("/v1/auth/accept-invite", { token, password })
  );
  return body.user;
}

export async function changePassword(
  oldPassword: string,
  newPassword: string
): Promise<void> {
  await jsonOrThrow(
    await jsonPost("/v1/auth/change-password", {
      old_password: oldPassword,
      new_password: newPassword,
    })
  );
}

// --------------------------------------------------------------------------- //
// admin
// --------------------------------------------------------------------------- //
export async function inviteUser(email: string, role: Role): Promise<InviteResult> {
  return jsonOrThrow(await jsonPost("/v1/auth/invite", { email, role }));
}

export async function listUsers(): Promise<AuthUser[]> {
  const body = await jsonOrThrow<{ data: AuthUser[] }>(
    await apiFetch("/v1/auth/users")
  );
  return body.data;
}

export async function setUserStatus(
  userId: string,
  status: UserStatus
): Promise<AuthUser> {
  const body = await jsonOrThrow<{ user: AuthUser }>(
    await jsonPost(`/v1/auth/users/${encodeURIComponent(userId)}/status`, { status })
  );
  return body.user;
}
