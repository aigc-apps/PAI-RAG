/** Copy text to the clipboard, resilient to insecure (HTTP) origins.
 *
 * `navigator.clipboard` only exists in a secure context (HTTPS or localhost);
 * when the app is served over plain HTTP on a LAN/server IP it is `undefined`,
 * so `navigator.clipboard.writeText(...)` throws and the copy silently fails.
 * We try the async API first, then fall back to a hidden-textarea +
 * `execCommand("copy")`. Returns true only when the copy actually landed. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to the legacy path
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-9999px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    ta.setSelectionRange(0, ta.value.length);
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}
