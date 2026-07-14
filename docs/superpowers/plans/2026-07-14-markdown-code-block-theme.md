# Themed Markdown Code Blocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render fenced Markdown code blocks with coherent light and dark syntax palettes and no line-sized dark backgrounds.

**Architecture:** Keep `CodeBlock` as the rendering boundary. Replace the fixed `oneDark` theme with a Prism theme whose token colors reference application CSS variables, so an existing `data-theme` switch updates syntax colors without React synchronization. Explicitly make the inner `code` surface transparent and keep source-code overflow horizontal.

**Tech Stack:** React, TypeScript, react-markdown, react-syntax-highlighter/Prism, Tailwind CSS, Vitest, Testing Library.

## Global Constraints

- The outer container, header, Prism `pre`, and inner `code` must form one theme-consistent surface.
- Source code preserves line structure and scrolls horizontally; plain output continues to wrap.
- The copy button keeps its localized accessible name, copied state, and visible keyboard focus.
- No new runtime dependency.

---

### Task 1: Adaptive fenced code block

**Files:**
- Modify: `frontend/src/components/Markdown.tsx`
- Modify: `frontend/src/index.css`
- Test: `frontend/src/components/__tests__/Markdown.test.tsx`

**Interfaces:**
- Consumes: existing `CodeBlock({ language, code })`, CSS theme selection via `:root[data-theme="dark"]`, and existing localization keys.
- Produces: `adaptiveCodeTheme`, a Prism style map backed by `--code-token-*` variables; fenced source blocks with a transparent inner `code` element.

- [ ] **Step 1: Write failing component tests**

Add tests that render a `shell` fence and assert that its inner `code` element has a transparent background, its scroll container preserves whitespace, and the copy control has the shared `focus-ring` class. Keep the existing plain-output test to guard wrapping behavior.

```tsx
it("renders source fences as a unified scrollable code surface", () => {
  const { container } = render(<Markdown content={"```shell\\necho $REGION\\n```"} />);
  const code = container.querySelector("pre code, div code");
  expect(code).toHaveStyle({ background: "transparent" });
  expect(code?.parentElement).toHaveClass("overflow-x-auto");
  expect(screen.getByRole("button", { name: "Copy code" })).toHaveClass("focus-ring");
});
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `npm test --prefix frontend -- --run frontend/src/components/__tests__/Markdown.test.tsx`

Expected: the new test fails because the fixed `oneDark` renderer does not make the inner code background transparent or expose the required overflow/focus contract.

- [ ] **Step 3: Add theme token variables**

Add light values under `:root` and dark values under `:root[data-theme="dark"]` for:

```css
--code-token-comment;
--code-token-punctuation;
--code-token-red;
--code-token-yellow;
--code-token-green;
--code-token-purple;
--code-token-blue;
--code-token-pink;
--code-token-orange;
```

Use restrained One Light-inspired values in light mode and One Dark-inspired values in dark mode. Also tune `--code-bg`, `--code-header`, `--code-border`, and `--code-text` as a matched surface in each theme.

- [ ] **Step 4: Implement adaptive Prism styling**

Remove the `oneDark` import and define a typed Prism style map in `Markdown.tsx` using the CSS variables. Pass it to `SyntaxHighlighter`; add `codeTagProps={{ style: { background: "transparent", display: "block", minWidth: "max-content" } }}`; add `overflowX: "auto"` and `maxHeight: "420px"` to the highlighter surface. Reduce header height to `h-8`, add `focus-ring` to the copy button, and retain the existing copy/copy-success behavior.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `npm test --prefix frontend -- --run frontend/src/components/__tests__/Markdown.test.tsx`

Expected: all Markdown component tests pass.

- [ ] **Step 6: Run frontend regression validation**

Run: `npm test --prefix frontend && npm run build --prefix frontend && git diff --check`

Expected: the complete frontend test suite passes, TypeScript and Vite production build succeed, and the diff has no whitespace errors.

- [ ] **Step 7: Commit the implementation**

```bash
git add frontend/src/components/Markdown.tsx frontend/src/index.css frontend/src/components/__tests__/Markdown.test.tsx docs/superpowers/plans/2026-07-14-markdown-code-block-theme.md
git commit -m "fix(ui): adapt markdown code blocks to theme"
```
