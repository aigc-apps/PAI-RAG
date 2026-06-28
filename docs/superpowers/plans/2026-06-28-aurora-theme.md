# Aurora Theme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `newfrontend/` its own modern identity — the "Aurora" theme (indigo/violet, soft/glassy, Inter, gentle motion) with light + dark modes and a toggle — restyled consistently across all components. Behavior unchanged.

**Architecture:** A CSS-variable design system in `index.css` (light on `:root`, dark on `:root[data-theme="dark"]`), a `useTheme` hook + `ThemeToggle` (localStorage + system default, no-flash init), then a token-driven restyle of every component + gentle motion. Pure presentation; existing tests stay green.

**Tech Stack:** Vite 6 + React 19 + TS + Tailwind v4 (CSS-variable tokens via `var(--…)` arbitrary values) + Vitest. From `newfrontend/`.

**Reference spec:** `docs/superpowers/specs/2026-06-28-aurora-theme-design.md`. Branch: `personal/yfei/agent-core`.

## Global Constraints

- **Presentation only** — no changes to reducer/store/hook/api logic or component props/handlers/aria-labels. All existing tests must stay green; `npm run build` clean.
- **All colors via tokens** (`var(--…)`) — no hardcoded hex in components (so dark mode + future themes work).
- **Motion gated** by `@media (prefers-reduced-motion: no-preference)`.
- Run `npm test` + `npm run build` at the end of each task.

---

## Key existing files

- `newfrontend/src/index.css` (current: minimal tokens from the previous task — to be replaced).
- `newfrontend/index.html` (Vite entry; add a no-flash theme script).
- Components: `App, Sidebar, ChatView, Composer, ModelSelector, MessageList, UserMessage, AssistantMessage, CollapsibleReasoning, ToolCall, Markdown, MessageControls`. `lib/cn.ts`, `lib/user.ts` (pattern for a localStorage helper). lucide-react (Sun, Moon, …), @radix-ui/react-collapsible.
- Tests in `src/**/__tests__/`. `App.test.tsx` mocks api/* + lib/user.

---

## Task 1: Aurora theme system + dark-mode toggle

**Files:** Replace `newfrontend/src/index.css`; modify `newfrontend/index.html`; create `src/lib/theme.ts` + `src/components/ThemeToggle.tsx`; wire `ThemeToggle` into `ChatView` top bar. Test: `src/lib/__tests__/theme.test.ts`, `src/components/__tests__/ThemeToggle.test.tsx`.

**Interfaces:** `getTheme(): "light"|"dark"` (stored or system); `setTheme(t)`, `toggleTheme()` (persist + apply `documentElement.dataset.theme`); `useTheme()` React hook returning `{ theme, toggle }`. `ThemeToggle` button (aria-label "Toggle theme", Sun/Moon).

- [ ] **Step 1: Write the failing tests**

`src/lib/__tests__/theme.test.ts`:
```ts
import { describe, it, expect, beforeEach, vi } from "vitest";
import { getTheme, setTheme, toggleTheme } from "../theme";

describe("theme", () => {
  beforeEach(() => { localStorage.clear(); document.documentElement.removeAttribute("data-theme"); });
  it("setTheme persists and applies data-theme", () => {
    setTheme("dark");
    expect(localStorage.getItem("agent-chat:theme")).toBe("dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    expect(getTheme()).toBe("dark");
  });
  it("toggleTheme flips light<->dark", () => {
    setTheme("light");
    expect(toggleTheme()).toBe("dark");
    expect(getTheme()).toBe("dark");
    expect(toggleTheme()).toBe("light");
  });
  it("defaults from system when unset", () => {
    vi.stubGlobal("matchMedia", (q: string) => ({ matches: q.includes("dark"), media: q, addEventListener() {}, removeEventListener() {} }));
    expect(getTheme()).toBe("dark");
  });
});
```

`src/components/__tests__/ThemeToggle.test.tsx`:
```tsx
import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeToggle } from "../ThemeToggle";

describe("ThemeToggle", () => {
  beforeEach(() => { localStorage.clear(); document.documentElement.removeAttribute("data-theme"); });
  it("toggles the document theme on click", async () => {
    render(<ThemeToggle />);
    const btn = screen.getByRole("button", { name: /toggle theme/i });
    const before = document.documentElement.getAttribute("data-theme");
    await userEvent.click(btn);
    expect(document.documentElement.getAttribute("data-theme")).not.toBe(before);
  });
});
```

- [ ] **Step 2: Run to verify failure** — `cd newfrontend && npm test -- "theme|ThemeToggle"` → FAIL.

- [ ] **Step 3: Implement `src/lib/theme.ts`**

```ts
export type Theme = "light" | "dark";
const KEY = "agent-chat:theme";

function systemTheme(): Theme {
  return typeof matchMedia !== "undefined" && matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark" : "light";
}
export function getTheme(): Theme {
  const stored = (typeof localStorage !== "undefined" && localStorage.getItem(KEY)) as Theme | null;
  return stored === "light" || stored === "dark" ? stored : systemTheme();
}
export function applyTheme(t: Theme): void {
  if (typeof document !== "undefined") document.documentElement.dataset.theme = t;
}
export function setTheme(t: Theme): Theme {
  try { localStorage.setItem(KEY, t); } catch { /* ignore */ }
  applyTheme(t);
  return t;
}
export function toggleTheme(): Theme {
  return setTheme(getTheme() === "dark" ? "light" : "dark");
}
```

Add a `useTheme` hook (in the same file or `src/hooks/useTheme.ts`):
```ts
import { useState, useEffect } from "react";
export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(getTheme);
  useEffect(() => { applyTheme(theme); }, [theme]);
  return { theme, toggle: () => setThemeState((t) => (t === "dark" ? "light" : "dark") as Theme & string && setTheme(t === "dark" ? "light" : "dark")) };
}
```
(If the one-liner toggle is awkward, write it plainly: compute `next`, `setTheme(next)`, `setThemeState(next)`.)

- [ ] **Step 4: Implement `src/components/ThemeToggle.tsx`**

```tsx
import { Sun, Moon } from "lucide-react";
import { useTheme } from "../lib/theme"; // or ../hooks/useTheme
export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button type="button" aria-label="Toggle theme" onClick={toggle}
      className="rounded-lg p-2 text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] transition-colors">
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}
```

- [ ] **Step 5: Replace `src/index.css`** with the Aurora tokens (light + dark), Inter import, base, scrollbars, motion keyframes:

```css
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");
@import "tailwindcss";

:root {
  --bg:#ffffff; --bg-elevated:#ffffff; --surface:#f7f7fb; --surface-2:#eceef5;
  --text:#15151a; --text-muted:#5b5d6b; --text-faint:#9a9cab;
  --border:#e8e8f0; --border-strong:#d7d8e3;
  --accent:#6d5efc; --accent-hover:#5b4ef0; --accent-soft:#eceafe; --accent-fg:#ffffff;
  --user-bubble:#f0f0f7; --tool-bg:#f7f7fb; --danger:#e5484d;
  --shadow-sm:0 1px 2px rgba(20,20,40,.06);
  --shadow:0 1px 2px rgba(20,20,40,.05), 0 10px 30px rgba(20,20,40,.08);
  --radius:14px; --radius-sm:10px; --radius-lg:22px;
  --accent-grad:linear-gradient(135deg,#6d5efc,#9b7bff);
}
:root[data-theme="dark"] {
  --bg:#0f1117; --bg-elevated:#171a23; --surface:#171a23; --surface-2:#1f2330;
  --text:#e7e8ee; --text-muted:#a4a7b5; --text-faint:#6b6e7e;
  --border:#262a36; --border-strong:#313644;
  --accent:#7c6cff; --accent-hover:#8f82ff; --accent-soft:#20203c; --accent-fg:#ffffff;
  --user-bubble:#232838; --tool-bg:#14171f; --danger:#ff6b6b;
  --shadow-sm:0 1px 2px rgba(0,0,0,.4);
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 12px 32px rgba(0,0,0,.4);
}

html, body, #root { height:100%; margin:0; }
* { box-sizing:border-box; }
body {
  background:var(--bg); color:var(--text);
  font-family:"Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size:16px; line-height:1.6; -webkit-font-smoothing:antialiased;
  transition:background-color .15s ease, color .15s ease;
}
.scrollbar-thin::-webkit-scrollbar { width:8px; height:8px; }
.scrollbar-thin::-webkit-scrollbar-thumb { background:var(--border-strong); border-radius:8px; }
.scrollbar-thin::-webkit-scrollbar-thumb:hover { background:var(--text-faint); }

@media (prefers-reduced-motion: no-preference) {
  @keyframes msg-in { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
  .animate-msg-in { animation:msg-in .25s ease both; }
  @keyframes shimmer { 0% { background-position:-200% 0; } 100% { background-position:200% 0; } }
  .shimmer-text {
    background:linear-gradient(90deg,var(--text-faint),var(--accent),var(--text-faint));
    background-size:200% 100%; -webkit-background-clip:text; background-clip:text;
    color:transparent; animation:shimmer 2s linear infinite;
  }
}
```

- [ ] **Step 6: No-flash init** in `newfrontend/index.html` — add to `<head>` before the module script:
```html
    <script>
      (function () {
        try {
          var t = localStorage.getItem("agent-chat:theme");
          if (t !== "light" && t !== "dark")
            t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
          document.documentElement.dataset.theme = t;
        } catch (e) {}
      })();
    </script>
```

- [ ] **Step 7: Wire `ThemeToggle` into the top bar** — in `ChatView.tsx`, add `<ThemeToggle />` to the top-bar row (next to the model selector). No other change.

- [ ] **Step 8: Run to verify pass + build** — `cd newfrontend && npm test && npm run build` → all pass.

- [ ] **Step 9: Commit**

```bash
git add newfrontend/src/index.css newfrontend/index.html newfrontend/src/lib/theme.ts newfrontend/src/components/ThemeToggle.tsx newfrontend/src/components/ChatView.tsx newfrontend/src/lib/__tests__/theme.test.ts newfrontend/src/components/__tests__/ThemeToggle.test.tsx
git commit -m "feat(newfrontend): Aurora theme system — light/dark tokens, Inter, motion, theme toggle"
```

---

## Task 2: Aurora shell restyle (sidebar, top bar, composer, empty state, brand)

**Files:** Modify `App.tsx`, `Sidebar.tsx`, `ChatView.tsx`, `Composer.tsx`, `ModelSelector.tsx`. Test: keep `App.test.tsx` green.

**Interfaces:** No behavior/prop changes. Apply the Aurora tokens + the visual direction below.

- [ ] **Step 1: Restyle to the Aurora spec** (write JSX to these contracts; behavior identical):

  **Brand mark** (a small inline component or inline JSX, reused in Sidebar header + empty state): an accent-gradient dot/badge (`bg-[var(--accent-grad)]` via `style={{background:"var(--accent-grad)"}}` since gradients aren't a Tailwind color token) + the wordmark "Aria" in `font-semibold`.

  **Sidebar.tsx:** `w-[264px] bg-[var(--surface)] border-r border-[var(--border)] flex flex-col`. Header row = brand mark. "New chat" = a pill `border border-[var(--border-strong)] rounded-[var(--radius-sm)]` ghost with `Plus`, hover `bg-[var(--surface-2)]`. List items: rounded, `text-[var(--text-muted)]`, hover `bg-[var(--surface-2)] text-[var(--text)]`; **active** (`selectedId===id`) = `bg-[var(--accent-soft)] text-[var(--text)] font-medium` (distinct from hover); `Trash2` shown on `group-hover`, `text-[var(--text-faint)] hover:text-[var(--danger)]`. Keep refresh/open/new/delete handlers.

  **ChatView top bar:** `h-14 flex items-center gap-1 px-3 border-b border-[var(--border)]` + `backdrop-blur bg-[var(--bg)]/80` (translucent). Left: sidebar toggle (`PanelLeft`, ghost icon button). Center/spacer. Right: `ModelSelector` + `ThemeToggle`. Keep the `resumeIfInterrupted` effect.

  **Empty state:** centered column (`flex-1 grid place-items-center`): the brand mark (larger), a heading `text-2xl font-semibold` "How can I help today?", then the `Composer` below it (`w-full max-w-2xl`). (Optional: 3 example-prompt chips that call `send(text)` — only if trivial; else skip.)

  **Composer.tsx:** centered `w-full max-w-3xl mx-auto`; container `bg-[var(--bg-elevated)] border border-[var(--border-strong)] rounded-[var(--radius-lg)] shadow-[var(--shadow)] px-3 py-2 flex items-end gap-2 focus-within:border-[var(--accent)] transition-colors`; textarea transparent auto-grow (rows 1→~6) placeholder "Message Aria…"; send button circular `h-8 w-8 rounded-full text-[var(--accent-fg)]` with `style={{background:"var(--accent-grad)"}}` + `hover:opacity-90`, disabled when empty (`opacity-40`); stop = square button while streaming. Keep Enter-send / Shift+Enter / clear / aria-labels.

  **ModelSelector.tsx:** keep `/v1/models` logic; restyle the `<select>` as `text-sm font-medium rounded-[var(--radius-sm)] px-2 py-1.5 text-[var(--text)] bg-transparent hover:bg-[var(--surface-2)] border border-transparent focus:border-[var(--border-strong)] outline-none`.

  **App.tsx:** `flex h-full bg-[var(--bg)]`; keep the sidebar toggle state.

- [ ] **Step 2: Keep `App.test.tsx` green** — adjust only if accessible names changed (keep aria-labels "Send"/"Stop"/"Model"/"Toggle theme"/"New chat" stable). The textbox + "New chat" + seeded-message assertions should still hold.

- [ ] **Step 3: Run to verify pass + build** — `cd newfrontend && npm test && npm run build` → pass.

- [ ] **Step 4: Commit**

```bash
git add newfrontend/src/components/App.tsx newfrontend/src/components/Sidebar.tsx newfrontend/src/components/ChatView.tsx newfrontend/src/components/Composer.tsx newfrontend/src/components/ModelSelector.tsx
git commit -m "feat(newfrontend): Aurora shell — branded sidebar, glassy top bar, gradient composer, empty state"
```

---

## Task 3: Aurora message / tool / reasoning restyle + motion

**Files:** Modify `MessageList.tsx`, `UserMessage.tsx`, `AssistantMessage.tsx`, `CollapsibleReasoning.tsx`, `ToolCall.tsx`, `Markdown.tsx`, `MessageControls.tsx`. Tests: keep `ToolCall`, `AssistantMessage`, `CollapsibleReasoning`, `MessageControls`, `Markdown` tests green.

**Interfaces:** No behavior changes. Apply Aurora + motion.

- [ ] **Step 1: Restyle to the Aurora spec:**

  **MessageList.tsx:** `mx-auto w-full max-w-3xl px-4 py-8 space-y-7`. Keep mapping + autoscroll + onRegenerate-to-last.

  **UserMessage.tsx:** right-aligned; bubble `bg-[var(--user-bubble)] text-[var(--text)] rounded-[var(--radius-lg)] px-4 py-2.5 max-w-[80%] whitespace-pre-wrap`; wrapper gets `animate-msg-in`.

  **AssistantMessage.tsx:** wrapper `flex gap-3 animate-msg-in`; a small avatar dot (`h-7 w-7 rounded-full shrink-0` with `style={{background:"var(--accent-grad)"}}`); then the content column = CollapsibleReasoning → toolCalls.map(ToolCall) → (failed→error box `bg-[var(--danger)]/10 text-[var(--danger)] rounded-[var(--radius)] px-3 py-2` | Markdown when text) → stopped/cancelled notes (`text-[var(--text-faint)] italic text-xs`) → MessageControls when completed||cancelled. Keep order + states + handlers.

  **CollapsibleReasoning.tsx:** disclosure trigger row: `ChevronRight` (rotates open) + label — while `status==="streaming"` the label "Thinking…" uses `shimmer-text`; when done "Thought" in `text-[var(--text-muted)]`. Body: `text-sm text-[var(--text-muted)] whitespace-pre-wrap border-l-2 border-[var(--accent)]/40 pl-3 mt-1`. Keep auto-open-streaming / auto-collapse-done / nothing-when-empty + the existing test contract (streaming→visible text, done→button).

  **ToolCall.tsx:** card `bg-[var(--tool-bg)] border border-[var(--border)] rounded-[var(--radius)] shadow-[var(--shadow-sm)] my-2 overflow-hidden`; header `<button>` (keep single button) `w-full flex items-center gap-2 px-3 py-2 text-sm`: status glyph (running `Loader2 animate-spin text-[var(--accent)]`; done `Check text-emerald-500`; error `X text-[var(--danger)]`) + `Wrench h-4 w-4 text-[var(--accent)]` + `name` (`font-medium`) + a muted status word (keep "error" for errors) + a chevron. Body (animated open): "Arguments" + `<pre>` (mono `text-xs bg-[var(--surface-2)] rounded-[var(--radius-sm)] p-2 whitespace-pre-wrap break-all`); "Result"/"Error" + scrollable `<pre>` (`max-h-64 overflow-auto`). Keep ToolUse import + the test contract.

  **Markdown.tsx:** Inter prose; `a` → `text-[var(--accent)] underline`; headings weight/size; lists/blockquote/table tokens; inline code `bg-[var(--surface-2)] rounded px-1`; fenced code keeps syntax highlighting + a copy button using `text-[var(--text-faint)] hover:text-[var(--text)]`. (Keep the `node`-prop-drop fix.)

  **MessageControls.tsx:** ghost icon buttons `text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] rounded-md p-1.5 transition-colors`; keep Copy/Regenerate aria-labels + behavior.

- [ ] **Step 2: Run to verify pass + build** — `cd newfrontend && npm test && npm run build` → pass.

- [ ] **Step 3: Commit**

```bash
git add newfrontend/src/components/MessageList.tsx newfrontend/src/components/UserMessage.tsx newfrontend/src/components/AssistantMessage.tsx newfrontend/src/components/CollapsibleReasoning.tsx newfrontend/src/components/ToolCall.tsx newfrontend/src/components/Markdown.tsx newfrontend/src/components/MessageControls.tsx
git commit -m "feat(newfrontend): Aurora messages — avatar, shimmer thinking, glassy tool cards, motion"
```

---

## Task 4: Verify both themes + polish

**Files:** any small fixes.

- [ ] **Step 1:** `cd newfrontend && npm test` → all green; `npm run build` → clean.
- [ ] **Step 2: Token-drift scan:** grep components for hardcoded colors (`text-gray-`, `bg-gray-`, `#`, `bg-white`, `text-white`, `bg-black`) and replace any stragglers with tokens (so dark mode is correct). Re-run tests + build.
- [ ] **Step 3: Reduced-motion + contrast sanity:** confirm animations are under `@media (prefers-reduced-motion: no-preference)`; confirm dark-mode text/border tokens are legible (the spec values are calibrated; fix any obvious low-contrast).
- [ ] **Step 4: Manual smoke (optional, needs the service):** run `newbackend` + `newfrontend` dev; toggle light/dark (no flash on reload); verify the empty state, streaming answer, shimmering "Thinking", tool cards (light+dark), sidebar active state, composer focus ring/gradient send.
- [ ] **Step 5: Commit any fixes**

```bash
git add -A newfrontend
git commit -m "chore(newfrontend): Aurora polish — token-drift sweep, reduced-motion + contrast checks"
```

---

## Self-Review (against the spec)

- **Aurora palette + light/dark tokens + Inter + motion** → Task 1 (`index.css`); toggle (`useTheme`/`ThemeToggle`/no-flash) → Task 1.
- **Own identity:** brand mark, accent-gradient composer/avatar, accent-soft active sidebar → Tasks 2–3.
- **Modern/clean surfaces:** glassy top bar, shadows, rounded, shimmer "Thinking", animated tool cards, message fade-in → Tasks 2–3.
- **Behavior unchanged / tests green:** only `useTheme`/`ThemeToggle` add logic (tested); all other changes are token/class swaps; aria-labels + handlers preserved → all tasks + Task 4 verify.
- **Token-only colors** (no drift → dark mode correct) → Task 4 sweep.

No behavior placeholders; the theme system is full code; component tasks specify exact tokens + structure + preserved contracts, implementer writes the JSX. Visual quality iterated with the user after first render.
