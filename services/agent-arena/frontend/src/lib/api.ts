const ACCESS_KEY_STORAGE_KEY = "agent-arena.access-key"

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

function stripTrailingSlash(value: string) {
  return value.replace(/\/+$/, "")
}

export function apiBaseUrl() {
  const configured = import.meta.env.VITE_ARENA_API_BASE?.trim()
  if (configured) return stripTrailingSlash(configured)

  const base = stripTrailingSlash(import.meta.env.BASE_URL || "/")
  return base ? `${base}/api` : "/api"
}

export function apiUrl(path: string) {
  const cleanPath = path.startsWith("/") ? path : `/${path}`
  return `${apiBaseUrl()}${cleanPath}`
}

export function getArenaAccessKey() {
  try {
    return window.localStorage.getItem(ACCESS_KEY_STORAGE_KEY) || ""
  } catch {
    return ""
  }
}

export function setArenaAccessKey(value: string) {
  try {
    const nextValue = value.trim()
    if (nextValue) {
      window.localStorage.setItem(ACCESS_KEY_STORAGE_KEY, nextValue)
    } else {
      window.localStorage.removeItem(ACCESS_KEY_STORAGE_KEY)
    }
  } catch {
    // Ignore storage failures; the next request will show the auth prompt again.
  }
}

export function clearArenaAccessKey() {
  setArenaAccessKey("")
}

export function apiHeaders(headers?: HeadersInit) {
  const nextHeaders = new Headers(headers)
  const accessKey = getArenaAccessKey()
  if (accessKey && !nextHeaders.has("Authorization")) {
    nextHeaders.set("Authorization", `Bearer ${accessKey}`)
  }
  return nextHeaders
}

export function apiFetch(path: string, init: RequestInit = {}) {
  return fetch(apiUrl(path), {
    ...init,
    headers: apiHeaders(init.headers),
  })
}

export async function readApiJson<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const detail =
      data && typeof data === "object" && "detail" in data
        ? String((data as { detail?: unknown }).detail || "")
        : ""
    throw new ApiError(response.status, detail || `HTTP ${response.status}`)
  }
  return data as T
}

export function isUnauthorizedError(error: unknown) {
  return error instanceof ApiError && error.status === 401
}
