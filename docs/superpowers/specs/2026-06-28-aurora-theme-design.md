# Aurora Theme — Modern Agent UI Design

**Date:** 2026-06-28
**Status:** Approved (user chose the "Aurora" direction)
**Branch:** `personal/yfei/agent-core`
**Builds on:** `newfrontend/` (the working ChatGPT-style UI with tool/reasoning/answer rendering). This is a **visual/theming iteration** — the data flow, reducer, stream client, and components already work and are tested; we give the UI its own modern identity.

## Problem

The current UI is functional but visually a ChatGPT clone (neutral grays, no identity). We want a **more modern, clean** look with its **own theme**: a distinctive indigo/violet brand ("Aurora"), **light + dark** modes with a toggle, soft/glassy surfaces, subtle gradients, and gentle motion — applied consistently across sidebar, messages, tool cards, reasoning, and composer.

## Decisions

1. **Aurora palette** — indigo/violet accent `#6d5efc`, soft neutral surfaces, glassy elevated cards, a signature accent gradient (`linear-gradient(135deg,#6d5efc,#9b7bff)`) for the brand mark / send button.
2. **Light + dark via CSS variables** on `:root` (light) and `:root[data-theme="dark"]` (dark). A `useTheme` hook (localStorage `agent-chat:theme`, default = system) + a `ThemeToggle` (Sun/Moon) in the top bar; a tiny inline script in `index.html` sets `data-theme` before paint (no flash).
3. **Inter font** (`@import` from Google Fonts, system fallback). 16px base, comfortable line-height.
4. **Gentle motion**, gated by `prefers-reduced-motion`: messages fade-in-up; the "Thinking…" label shimmers while streaming; tool cards animate open; hover/focus transitions.
5. **Behavior unchanged** — purely presentational; all existing component tests stay green (aria-labels, handlers preserved). The only new logic is the theme toggle (small, tested).

## Design tokens (`index.css`)

```
LIGHT (:root)                         DARK (:root[data-theme="dark"])
--bg            #ffffff               #0f1117
--bg-elevated   #ffffff               #171a23
--surface       #f7f7fb               #171a23   (sidebar, cards)
--surface-2     #eceef5               #1f2330   (hover/active)
--text          #15151a               #e7e8ee
--text-muted    #5b5d6b               #a4a7b5
--text-faint    #9a9cab               #6b6e7e
--border        #e8e8f0               #262a36
--border-strong #d7d8e3               #313644
--accent        #6d5efc               #7c6cff
--accent-hover  #5b4ef0               #8f82ff
--accent-soft   #eceafe               #20203c   (accent-tinted bg)
--accent-fg     #ffffff               #ffffff
--user-bubble   #f0f0f7               #232838
--tool-bg       #f7f7fb               #14171f
--danger        #e5484d               #ff6b6b
--shadow-sm     0 1px 2px rgba(20,20,40,.06)
--shadow        0 1px 2px rgba(20,20,40,.05), 0 10px 30px rgba(20,20,40,.08)   (dark: rgba(0,0,0,.45))
--radius        14px       --radius-sm 10px       --radius-lg 22px
--accent-grad   linear-gradient(135deg,#6d5efc,#9b7bff)
```

Base: Inter font; `body{background:var(--bg);color:var(--text)}`; thin themed scrollbars; `*{box-sizing:border-box}`; smooth `color/background` transitions on theme switch (~150ms). Motion keyframes `msg-in` (opacity+translateY) and `shimmer`; all animations wrapped in `@media (prefers-reduced-motion: no-preference)`.

## Visual direction per area

- **Brand mark:** "Aria" wordmark with an accent-gradient dot/logo, in the sidebar header and empty state.
- **Sidebar:** `--surface` bg, subtle right border; "New chat" = accent-tinted/outline pill with `+`; list items rounded, hover `--surface-2`, **active** = `--accent-soft` + accent left-edge or accent text + medium weight (distinct from hover); trash on hover.
- **Top bar:** minimal, translucent (`backdrop-blur`) with a hairline bottom border; sidebar toggle, model selector, **ThemeToggle** (Sun/Moon), spacer.
- **Messages:** centered `max-w-3xl`; **user** = right bubble tinted `--user-bubble` (subtle), rounded-2xl; **assistant** = full-width with a small accent-gradient avatar dot; messages **fade-in-up** on mount.
- **Thinking:** a muted disclosure; while streaming the "Thinking…" label **shimmers** (accent-tinted moving gradient); auto-collapse on done; body in a soft `--accent-soft`/muted block with an accent left border.
- **Tool cards:** glassy `--tool-bg` card, `--radius`, `--shadow-sm`, accent-tinted header icon; status glyph (spinner/check/x) colored (accent/green/danger); animated expand of args/result.
- **Composer:** a `--bg-elevated` rounded-`--radius-lg` pill with `--shadow`, focus ring in `--accent`; circular send button filled with `--accent-grad` (subtle glow on hover); square stop while streaming.
- **Empty state:** centered brand mark + "How can I help today?" + the composer mid-screen; maybe 3–4 example prompt chips (optional).
- **Markdown:** Inter prose; links in `--accent`; code blocks keep syntax highlighting (a dark theme that fits both modes), code copy button using tokens; inline code on `--surface-2`.

## Testing

- **`useTheme`:** defaults to system when unset; `setTheme`/`toggle` persist to localStorage and set `document.documentElement.dataset.theme`.
- **`ThemeToggle`:** renders a button (aria-label "Toggle theme"); clicking flips light↔dark.
- **No regressions:** all existing component tests stay green (handlers + aria-labels unchanged); `App` smoke still passes; `npm run build` clean. Visual quality is iterated with the user.

## Out of scope
- Per-conversation themes; theme customization UI; multiple accents. (Tokens make these cheap later.)
- Reworking message/tool DATA (already done) — this is styling only.

## Sequencing
One spec, one plan, subagent-driven: (1) theme system + dark toggle, (2) Aurora shell restyle, (3) Aurora message/tool/reasoning restyle + motion, (4) verify/polish both themes.
