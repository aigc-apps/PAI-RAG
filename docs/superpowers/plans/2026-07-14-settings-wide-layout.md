# Settings Wide Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the settings workspace to a balanced 1440px desktop layout and improve Agent/knowledge content density without changing behavior.

**Architecture:** Add reusable settings-shell CSS for the 1440px grid, sticky navigation, reading-width panels, and responsive collapse. Keep component-specific density rules in `SettingsView.tsx` and `KnowledgeView.tsx`, with semantic test hooks for layout regression coverage.

**Tech Stack:** React 19, TypeScript, Tailwind utility classes, shared CSS variables, Vitest, Testing Library, Vite.

## Global Constraints

- Settings desktop maximum width is 1440px with a 184px navigation column.
- Dense workspaces use the full content column; Persona, default persona, and YAML are capped near 960px.
- Desktop navigation is sticky; below 1024px it becomes a horizontal scrolling navigation row.
- Knowledge document tables preserve readable column widths and use horizontal scrolling when necessary.
- Existing routes, APIs, data behavior, dialogs, drawers, themes, and chat width remain unchanged.
- Use existing theme variables and component primitives; add no dependency or new brand color.

---

### Task 1: Build the responsive settings shell

**Files:**
- Modify: `frontend/src/components/__tests__/SettingsView.test.tsx`
- Modify: `frontend/src/components/SettingsView.tsx`
- Modify: `frontend/src/index.css`

**Interfaces:**
- Produces `.settings-layout`, `.settings-nav`, and `.settings-reading-pane` CSS classes.
- `SettingsView` marks its main grid with `data-testid="settings-layout"` and navigation with `data-testid="settings-navigation"`.

- [ ] **Step 1: Write the failing shell test**

```typescript
it("uses the balanced wide settings workspace", () => {
  render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);
  expect(screen.getByTestId("settings-layout")).toHaveClass("settings-layout");
  expect(screen.getByTestId("settings-navigation")).toHaveClass("settings-nav");
});
```

Extend the test by clicking `默认人格` and asserting the main section contains `settings-reading-pane`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cd frontend
npm test -- --run src/components/__tests__/SettingsView.test.tsx -t "balanced wide settings workspace"
```

Expected: FAIL because the test IDs and settings CSS classes do not exist.

- [ ] **Step 3: Implement the shell CSS and component hooks**

Add to `index.css`:

```css
.settings-layout {
  width: 100%;
  max-width: 1440px;
  margin-inline: auto;
  padding: 24px 32px 64px;
  display: grid;
  grid-template-columns: 184px minmax(0, 1fr);
  gap: 32px;
  align-items: start;
}

.settings-nav { position: sticky; top: 24px; align-self: start; }
.settings-reading-pane { width: 100%; max-width: 960px; }

@media (max-width: 1279px) {
  .settings-layout { padding-inline: 24px; gap: 24px; }
}

@media (max-width: 1023px) {
  .settings-layout { grid-template-columns: minmax(0, 1fr); }
  .settings-nav { position: static; display: flex; gap: 4px; overflow-x: auto; }
}

@media (max-width: 767px) {
  .settings-layout { padding: 20px 16px 48px; }
}
```

Replace the current `<main>` grid utilities with `settings-layout overflow-y-auto`. Add `settings-nav` to `<aside>`. Compute:

```typescript
const readingPane = tab === "org-persona" || tab === "yaml";
```

and add `settings-reading-pane` conditionally to the content `<section>`.

- [ ] **Step 4: Verify GREEN**

```bash
cd frontend
npm test -- --run src/components/__tests__/SettingsView.test.tsx -t "balanced wide settings workspace"
```

Expected: PASS.

### Task 2: Stabilize Agent summary-card density

**Files:**
- Modify: `frontend/src/components/__tests__/SettingsView.test.tsx`
- Modify: `frontend/src/components/SettingsView.tsx`

**Interfaces:**
- `PreviewCard` emits a full-height card.
- Agent summary grid uses two columns at medium widths and three at extra-large widths.

- [ ] **Step 1: Write the failing Agent-card test**

```typescript
it("keeps agent summary cards equal height in the wide workspace", () => {
  render(<SettingsView doc={baseDoc} onBack={vi.fn()} />);
  const grid = screen.getByTestId("agent-summary-grid");
  expect(grid).toHaveClass("md:grid-cols-2", "xl:grid-cols-3");
  for (const card of screen.getAllByTestId("agent-summary-card")) {
    expect(card).toHaveClass("h-full");
  }
});
```

- [ ] **Step 2: Run and verify RED**

```bash
cd frontend
npm test -- --run src/components/__tests__/SettingsView.test.tsx -t "summary cards equal height"
```

Expected: FAIL because summary test IDs and the responsive grid classes do not exist.

- [ ] **Step 3: Implement equal-height cards**

Add `data-testid="agent-summary-card"` and `h-full` to `PreviewCard`. Change the capability grid to:

```tsx
<div data-testid="agent-summary-grid" className="grid items-stretch gap-4 md:grid-cols-2 xl:grid-cols-3">
```

Keep Persona outside the three-card grid and retain existing dialogs and content truncation.

- [ ] **Step 4: Verify GREEN**

```bash
cd frontend
npm test -- --run src/components/__tests__/SettingsView.test.tsx -t "summary cards equal height"
```

Expected: PASS.

### Task 3: Give the embedded knowledge workspace a stable table layout

**Files:**
- Modify: `frontend/src/components/__tests__/KnowledgeView.test.tsx`
- Modify: `frontend/src/components/KnowledgeView.tsx`

**Interfaces:**
- Embedded detail uses `knowledge-embedded-detail` and `knowledge-tabbar` layout hooks.
- Documents table uses `data-testid="knowledge-documents-table"`, a minimum width, and explicit column sizing.

- [ ] **Step 1: Write failing knowledge-layout tests**

Mock `listKnowledgeDocuments` with one long-title document, render embedded `KnowledgeView` on the `files` tab, then assert:

```typescript
expect(screen.getByTestId("knowledge-embedded-detail")).toHaveClass("min-w-0");
expect(screen.getByTestId("knowledge-tabbar")).toHaveClass("overflow-x-auto");
expect(await screen.findByTestId("knowledge-documents-table")).toHaveClass("min-w-[940px]");
expect(screen.getByTestId("knowledge-document-title-cell")).toHaveClass("min-w-[300px]");
```

- [ ] **Step 2: Run and verify RED**

```bash
cd frontend
npm test -- --run src/components/__tests__/KnowledgeView.test.tsx -t "stable wide document table"
```

Expected: FAIL because the table hooks and stable sizing classes do not exist.

- [ ] **Step 3: Implement the embedded toolbar and table sizing**

Add `data-testid="knowledge-tabbar"` and `overflow-x-auto` to the tab bar. Mark the embedded root `data-testid="knowledge-embedded-detail"` and `min-w-0`.

Change the table to `w-full min-w-[940px] table-fixed`. Add a `<colgroup>`:

```tsx
<colgroup>
  <col />
  <col className="w-[84px]" />
  <col className="w-[92px]" />
  <col className="w-[64px]" />
  <col className="w-[180px]" />
  <col className="w-[104px]" />
  <col className="w-[88px]" />
</colgroup>
```

Mark the title cell `data-testid="knowledge-document-title-cell"` with `min-w-[300px]`; make its URL use `max-w-full truncate`. Keep the outer `overflow-x-auto`; use `whitespace-nowrap` for source, status, chunk, time, and action cells. Keep the tag cell wrapping within its fixed column.

- [ ] **Step 4: Verify GREEN and run the full frontend suite**

```bash
cd frontend
npm test -- --run src/components/__tests__/KnowledgeView.test.tsx
npm test -- --run
npm run build
```

Expected: all tests pass and the production build succeeds.

### Task 4: Final scope and visual-quality verification

**Files:**
- Verify: `frontend/src/components/SettingsView.tsx`
- Verify: `frontend/src/components/KnowledgeView.tsx`
- Verify: `frontend/src/index.css`

- [ ] **Step 1: Run formatting and diff checks**

```bash
git diff --check
cd frontend && npm run build
```

Expected: no whitespace errors and a successful TypeScript/Vite build.

- [ ] **Step 2: Inspect responsive layout classes**

```bash
rg -n "settings-layout|settings-nav|settings-reading-pane|agent-summary-grid|knowledge-documents-table|min-w-\[940px\]" frontend/src
```

Expected: every layout contract is present in its intended component or stylesheet.

- [ ] **Step 3: Commit implementation and plan**

```bash
git add frontend/src/components/SettingsView.tsx \
  frontend/src/components/KnowledgeView.tsx \
  frontend/src/components/__tests__/SettingsView.test.tsx \
  frontend/src/components/__tests__/KnowledgeView.test.tsx \
  frontend/src/index.css \
  docs/superpowers/plans/2026-07-14-settings-wide-layout.md
git commit -m "feat(ui): widen and refine settings workspace"
```
