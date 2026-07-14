# Tool Call Status and Duration Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace redundant visible per-row status text with an accessible status dot and align tool durations in a fixed right-side column.

**Architecture:** Keep the existing `ToolUse` status and duration data unchanged. Update only `ToolCall` rendering so the flexible argument summary yields space to a fixed duration column and the existing error detail remains authoritative.

**Tech Stack:** React 19, TypeScript 5.6, Tailwind CSS 4, Vitest 3, Testing Library.

## Global Constraints

- Status colors remain accent/pulsing for running, green for done, and red for error.
- Visible `Running`, `Done`, and `Failed` row text is removed.
- The status dot exposes the localized state through `role="img"` and `aria-label`.
- Duration calculation and formatting do not change.
- The duration column is fixed-width, right-aligned, tabular, and immediately precedes the chevron.
- Running or duration-less tools retain an empty fixed-width column.
- Error rows remain red-tinted, auto-expanded, and show the existing failure card.

---

### Task 1: Accessible Status Dot and Right-Aligned Duration

**Files:**
- Modify: `frontend/src/components/__tests__/ToolCall.test.tsx`
- Modify: `frontend/src/components/ToolCall.tsx`

**Interfaces:**
- Consumes: existing `ToolUse.status`, `ToolUse.durationMs`, and i18n status keys.
- Produces: an accessible leading status indicator and stable right-side duration column.

- [ ] **Step 1: Write failing component tests**

Extend `ToolCall.test.tsx` with:

```tsx
it("uses the localized dot as the only visible completed status", () => {
  render(
    <ToolCall
      tool={{
        id: "c1",
        name: "shell",
        arguments: '{"command":"pwd"}',
        status: "done",
        output: "",
        durationMs: 2100,
      }}
    />,
  );

  expect(screen.getByRole("img", { name: "完成" })).toBeInTheDocument();
  expect(screen.queryByText("完成")).not.toBeInTheDocument();
});

it("aligns duration in a fixed right-side column", () => {
  render(
    <ToolCall
      tool={{
        id: "c1",
        name: "shell",
        arguments: '{"command":"pwd"}',
        status: "done",
        output: "",
        durationMs: 2100,
      }}
    />,
  );

  const duration = screen.getByText("2.1s");
  expect(duration).toHaveClass("ml-auto", "w-14", "text-right", "tabular-nums");
  expect(duration.nextElementSibling).toHaveClass("lucide-chevron-right");
});

it("keeps an empty duration column for running tools", () => {
  render(
    <ToolCall
      tool={{
        id: "c1",
        name: "web_search",
        arguments: '{"query":"PAI-RAG"}',
        status: "running",
      }}
    />,
  );

  expect(screen.getByRole("img", { name: "运行中" })).toBeInTheDocument();
  expect(screen.queryByText("运行中")).not.toBeInTheDocument();
  expect(screen.getByTestId("tool-duration")).toBeEmptyDOMElement();
});
```

Strengthen the existing error test:

```tsx
expect(screen.getByRole("img", { name: "失败" })).toBeInTheDocument();
expect(screen.queryByText("失败")).not.toBeInTheDocument();
expect(screen.getByRole("button")).toHaveAttribute("aria-expanded", "true");
expect(screen.getByText("工具执行失败")).toBeInTheDocument();
expect(screen.getByText("boom")).toBeInTheDocument();
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
cd frontend && npm test -- src/components/__tests__/ToolCall.test.tsx
```

Expected: FAIL because the dot has no accessible name, visible status text remains, and the duration has no fixed right-side column.

- [ ] **Step 3: Implement the minimal rendering change**

Change `StatusDot` to accept the localized label and expose it accessibly:

```tsx
function StatusDot({
  status,
  label,
}: {
  status: ToolUse["status"];
  label: string;
}) {
  const common = "h-2 w-2 shrink-0 rounded-full";
  if (status === "running")
    return (
      <span
        role="img"
        aria-label={label}
        className={`${common} bg-[var(--accent)] pulse-dot`}
      />
    );
  if (status === "done")
    return (
      <span
        role="img"
        aria-label={label}
        className={`${common} bg-[var(--success)]`}
      />
    );
  return (
    <span
      role="img"
      aria-label={label}
      className={`${common} bg-[var(--danger)]`}
    />
  );
}
```

Render it with:

```tsx
<StatusDot status={tool.status} label={t(STATUS_KEY[tool.status])} />
```

Delete `STATUS_TONE` and the visible status `<span>`. Replace the conditional duration with an always-present column:

```tsx
<span
  data-testid="tool-duration"
  className="ml-auto w-14 shrink-0 text-right font-mono text-xs tabular-nums text-[var(--text-faint)]"
>
  {tool.durationMs != null && tool.status !== "running"
    ? formatDuration(tool.durationMs)
    : null}
</span>
```

Remove `ml-auto` from the chevron class so the duration column owns right-side spacing. Do not change expanded content or error auto-expansion.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same Vitest command from Step 2.

Expected: all `ToolCall` tests PASS.

- [ ] **Step 5: Run complete frontend verification**

```bash
cd frontend && npm test && npm run build
```

Expected: all frontend tests and the production build PASS. Existing unrelated React `act(...)` and bundle-size warnings may remain.

- [ ] **Step 6: Commit and push**

```bash
git add frontend/src/components/ToolCall.tsx frontend/src/components/__tests__/ToolCall.test.tsx
git commit -m "refactor(ui): align tool duration and status"
git push origin personal/yfei/agent-core
```
