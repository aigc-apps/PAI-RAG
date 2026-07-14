# Sidebar Settings Shortcut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an administrator-only Ghost settings button to the right side of the PAI-Loop sidebar logo.

**Architecture:** Reuse the existing optional `Sidebar.onOpenSettings` callback as both the visibility boundary and click handler. Render the shortcut in the existing brand row without adding auth state, API calls, or routes; `ChatPage` remains responsible for passing the callback only to administrators.

**Tech Stack:** React 19, TypeScript, Lucide React, Tailwind CSS utilities, Vitest, Testing Library.

## Global Constraints

- The shortcut appears only when `onOpenSettings` is provided.
- The button is a 32×32px Ghost gear icon aligned to the right of the PAI-Loop brand.
- The localized Settings text is used for `title` and `aria-label`.
- Clicking invokes the existing callback, which navigates administrators to `/settings/agents`.
- The account-menu Settings entry and server-backed route guard remain unchanged.
- Add no role lookup, state, API call, route, dependency, or brand color.

---

### Task 1: Add the administrator sidebar shortcut

**Files:**
- Modify: `frontend/src/components/__tests__/Sidebar.test.tsx`
- Modify: `frontend/src/components/Sidebar.tsx`

**Interfaces:**
- Consumes: `Sidebar({ onOpenSettings?: () => void, onOpenUsers?: () => void })`.
- Produces: A header button named by `t("userMenu.settings")` that invokes `onOpenSettings` exactly once when clicked.

- [ ] **Step 1: Extend the Sidebar test renderer**

Update the helper so tests can provide the existing callback:

```tsx
function renderSidebar(path = "/", onOpenSettings?: () => void) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Sidebar onOpenSettings={onOpenSettings} />
      <div data-testid="location">{<Location />}</div>
    </MemoryRouter>,
  );
}
```

- [ ] **Step 2: Write failing visibility and click tests**

Add focused behavior tests:

```tsx
it("hides the header Settings shortcut without an admin callback", () => {
  renderSidebar();
  expect(screen.queryByRole("button", { name: "设置" })).not.toBeInTheDocument();
});

it("opens Settings from the header shortcut when enabled", async () => {
  const user = userEvent.setup();
  const onOpenSettings = vi.fn();
  renderSidebar("/", onOpenSettings);

  const shortcut = screen.getByRole("button", { name: "设置" });
  expect(shortcut).toHaveAttribute("title", "设置");
  await user.click(shortcut);
  expect(onOpenSettings).toHaveBeenCalledTimes(1);
});
```

- [ ] **Step 3: Run the focused test and verify RED**

Run:

```bash
cd frontend
npm test -- --run src/components/__tests__/Sidebar.test.tsx -t "header Settings shortcut|opens Settings from the header shortcut"
```

Expected: the visibility test passes and the enabled shortcut test fails because the brand row does not render a Settings button.

- [ ] **Step 4: Implement the Ghost button**

Import `Settings2` from `lucide-react`, allow the brand component to fill available width, and conditionally render the shortcut in the header:

```tsx
import { Plus, Settings2, Trash2 } from "lucide-react";

<div className="flex h-[var(--header-h)] flex-shrink-0 items-center gap-2 border-b border-[var(--border)] bg-[var(--bg-elevated)]/70 px-4">
  <div className="min-w-0 flex-1">
    <BrandMark />
  </div>
  {onOpenSettings && (
    <button
      type="button"
      onClick={onOpenSettings}
      aria-label={t("userMenu.settings")}
      title={t("userMenu.settings")}
      className="focus-ring grid h-8 w-8 flex-shrink-0 place-items-center rounded-[var(--radius)] text-[var(--text-muted)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--text)]"
    >
      <Settings2 className="h-4 w-4" />
    </button>
  )}
</div>
```

- [ ] **Step 5: Verify GREEN and regressions**

Run:

```bash
cd frontend
npm test -- --run src/components/__tests__/Sidebar.test.tsx src/components/__tests__/AppGuards.test.tsx src/components/__tests__/UserMenu.test.tsx
npm test -- --run
npm run build
```

Expected: all focused and full frontend tests pass; TypeScript and Vite production build succeed. Existing account-menu and route-guard tests remain green.

- [ ] **Step 6: Check scope and commit**

Run:

```bash
git diff --check
git status --short
git add frontend/src/components/Sidebar.tsx \
  frontend/src/components/__tests__/Sidebar.test.tsx \
  docs/superpowers/plans/2026-07-14-sidebar-settings-shortcut.md
git commit -m "feat(ui): add admin settings shortcut"
```

Expected: only the Sidebar component, its tests, and this implementation plan are included in the commit.
