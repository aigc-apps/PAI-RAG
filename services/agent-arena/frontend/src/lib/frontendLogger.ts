import { apiHeaders, apiUrl, getArenaAccessKey } from "@/lib/api"

type FrontendLogPayload = {
  level?: "debug" | "info" | "warning" | "error"
  message: string
  stack?: string
  component_stack?: string
  source?: string
  payload?: Record<string, unknown>
}

function normalizeReason(reason: unknown): { message: string; stack?: string } {
  if (reason instanceof Error) {
    return { message: reason.message, stack: reason.stack }
  }
  if (typeof reason === "string") {
    return { message: reason }
  }
  try {
    return { message: JSON.stringify(reason) }
  } catch {
    return { message: String(reason) }
  }
}

export function reportFrontendLog(payload: FrontendLogPayload) {
  const body = {
    level: payload.level || "error",
    message: payload.message,
    stack: payload.stack,
    component_stack: payload.component_stack,
    source: payload.source || "frontend",
    payload: payload.payload || {},
    url: window.location.href,
    user_agent: navigator.userAgent,
    timestamp: new Date().toISOString(),
  }

  const encoded = JSON.stringify(body)
  const accessKey = getArenaAccessKey()
  try {
    if (!accessKey && navigator.sendBeacon) {
      const blob = new Blob([encoded], { type: "application/json" })
      if (navigator.sendBeacon(apiUrl("/frontend-log"), blob)) return
    }
  } catch {
    // Fall through to fetch.
  }

  void fetch(apiUrl("/frontend-log"), {
    method: "POST",
    headers: apiHeaders({ "Content-Type": "application/json" }),
    body: encoded,
    keepalive: true,
  }).catch(() => {
    // Logging must never create a second visible failure.
  })
}

export function installGlobalFrontendLogging() {
  window.addEventListener("error", (event) => {
    reportFrontendLog({
      level: "error",
      source: "window.error",
      message: event.message || "Uncaught frontend error",
      stack: event.error instanceof Error ? event.error.stack : undefined,
      payload: {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
      },
    })
  })

  window.addEventListener("unhandledrejection", (event) => {
    const normalized = normalizeReason(event.reason)
    reportFrontendLog({
      level: "error",
      source: "window.unhandledrejection",
      message: normalized.message || "Unhandled promise rejection",
      stack: normalized.stack,
    })
  })
}

export function normalizeError(error: unknown) {
  return normalizeReason(error)
}
