# Localized Tool Call Title Summaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a localized display name, raw tool name, and concise primary-argument summary in every built-in tool-call title.

**Architecture:** Add one pure frontend display-policy module that owns built-in tool labels and argument-summary formatters. `ToolCall` consumes that policy and renders three visual levels without changing the tool event model or expanded details.

**Tech Stack:** React 19, TypeScript 5.6, Tailwind CSS 4, Vitest 3, Testing Library.

## Global Constraints

- This is a frontend-only presentation change; do not change backend schemas, events, persistence, or serialization.
- Title format is `Localized name (raw_tool_name) · argument summary`.
- Localized names must support both existing languages, Chinese and English.
- Only the configured important fields may appear in summaries; do not dump arbitrary objects or arrays.
- Keep titles on one line; truncate only the summary and expose its full normalized value through native hover text.
- Unknown tools keep the current raw-name-only display.
- Expanded arguments and results remain unchanged.

---

### Task 1: Built-in Tool Display Policy

**Files:**
- Create: `frontend/src/lib/toolCallDisplay.ts`
- Create: `frontend/src/lib/__tests__/toolCallDisplay.test.ts`
- Modify: `frontend/src/i18n/en.ts`
- Modify: `frontend/src/i18n/zh.ts`

**Interfaces:**
- Consumes: raw `toolName: string` and serialized `rawArguments: string` from `ToolUse`.
- Produces: `getToolCallDisplay(toolName: string, rawArguments: string): ToolCallDisplay`, where `ToolCallDisplay` is `{ labelKey?: MessageKey; summary?: string }`.

- [ ] **Step 1: Write failing display-policy tests**

Create `frontend/src/lib/__tests__/toolCallDisplay.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { getToolCallDisplay } from "../toolCallDisplay";

describe("getToolCallDisplay", () => {
  it.each([
    ["shell", { command: "ls /opt/code" }, "tool.name.shell", "ls /opt/code"],
    ["code_interpreter", { language: "python", code: "print(1)\nprint(2)" }, "tool.name.codeInterpreter", "python · print(1)"],
    ["web_search", { query: "PAI-RAG" }, "tool.name.webSearch", "PAI-RAG"],
    ["web_fetch", { url: "https://example.com" }, "tool.name.webFetch", "https://example.com"],
    ["knowledge_search", { query: "向量检索" }, "tool.name.knowledgeSearch", "向量检索"],
    ["knowledge_find", { query: "ERR_42" }, "tool.name.knowledgeFind", "ERR_42"],
    ["knowledge_read", { document_id: "doc_1", chunk_id: "chk_1" }, "tool.name.knowledgeRead", "doc_1"],
    ["knowledge_list", {}, "tool.name.knowledgeList", undefined],
    ["current_datetime", {}, "tool.name.currentDatetime", undefined],
    ["load_skill", { skill_id: "skill.pdf" }, "tool.name.loadSkill", "skill.pdf"],
    ["enable_skill_for_agent", { skill_id: "skill.pdf", enabled: false }, "tool.name.enableSkillForAgent", "skill.pdf"],
    ["read_skill_resource", { skill_id: "skill.pdf", path: "references/a.md" }, "tool.name.readSkillResource", "skill.pdf · references/a.md"],
    ["install_skill", { source: { type: "git", url: "https://github.com/acme/skills", path: "pdf" } }, "tool.name.installSkill", "git · https://github.com/acme/skills · pdf"],
    ["publish_artifact", { name: "report.pdf", path: "/mnt/user/report.pdf" }, "tool.name.publishArtifact", "report.pdf"],
    ["spawn_subagent", { agent_id: "explore", task: "inspect the repository" }, "tool.name.spawnSubagent", "explore · inspect the repository"],
    ["read_handle", { handle: "store://tool/call_1" }, "tool.name.readHandle", "store://tool/call_1"],
  ])("formats %s", (name, args, labelKey, summary) => {
    expect(getToolCallDisplay(name, JSON.stringify(args))).toEqual({ labelKey, summary });
  });

  it("falls back from document_id to chunk_id", () => {
    expect(getToolCallDisplay("knowledge_read", '{"chunk_id":"chk_2"}').summary).toBe("chk_2");
  });

  it("normalizes multiline and repeated whitespace", () => {
    expect(getToolCallDisplay("shell", '{"command":"git  status\\n --short"}').summary).toBe("git status --short");
  });

  it.each(["not-json", "[]", "null", "{}"])("keeps a known label without a usable summary for %s", (raw) => {
    expect(getToolCallDisplay("shell", raw)).toEqual({ labelKey: "tool.name.shell", summary: undefined });
  });

  it("preserves unknown tools through an empty policy result", () => {
    expect(getToolCallDisplay("custom_tool", '{"query":"secret"}')).toEqual({});
  });
});
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd frontend && npm test -- src/lib/__tests__/toolCallDisplay.test.ts
```

Expected: FAIL because `../toolCallDisplay` does not exist.

- [ ] **Step 3: Add matching Chinese and English display-name keys**

Insert the following entries beside the existing `tool.*` strings in `frontend/src/i18n/en.ts`:

```ts
  "tool.name.shell": "Run command",
  "tool.name.codeInterpreter": "Run code",
  "tool.name.webSearch": "Search the web",
  "tool.name.webFetch": "Fetch webpage",
  "tool.name.knowledgeSearch": "Search knowledge",
  "tool.name.knowledgeFind": "Find in knowledge",
  "tool.name.knowledgeRead": "Read knowledge document",
  "tool.name.knowledgeList": "List knowledge bases",
  "tool.name.currentDatetime": "Get current time",
  "tool.name.loadSkill": "Load skill",
  "tool.name.enableSkillForAgent": "Configure agent skill",
  "tool.name.readSkillResource": "Read skill resource",
  "tool.name.installSkill": "Install skill",
  "tool.name.publishArtifact": "Publish artifact",
  "tool.name.spawnSubagent": "Start subagent",
  "tool.name.readHandle": "Read stored result",
```

Insert the matching entries in `frontend/src/i18n/zh.ts`:

```ts
  "tool.name.shell": "执行命令",
  "tool.name.codeInterpreter": "运行代码",
  "tool.name.webSearch": "搜索网页",
  "tool.name.webFetch": "读取网页",
  "tool.name.knowledgeSearch": "搜索知识库",
  "tool.name.knowledgeFind": "精确查找知识",
  "tool.name.knowledgeRead": "读取知识文档",
  "tool.name.knowledgeList": "列出知识库",
  "tool.name.currentDatetime": "获取当前时间",
  "tool.name.loadSkill": "加载技能",
  "tool.name.enableSkillForAgent": "配置 Agent 技能",
  "tool.name.readSkillResource": "读取技能资源",
  "tool.name.installSkill": "安装技能",
  "tool.name.publishArtifact": "发布文件",
  "tool.name.spawnSubagent": "启动子 Agent",
  "tool.name.readHandle": "读取已存结果",
```

- [ ] **Step 4: Implement the pure display policy**

Create `frontend/src/lib/toolCallDisplay.ts`:

```ts
import type { MessageKey } from "../i18n";

type ArgumentsObject = Record<string, unknown>;
type SummaryFormatter = (args: ArgumentsObject) => string | undefined;

interface ToolDisplayConfig {
  labelKey: MessageKey;
  summary?: SummaryFormatter;
}

export interface ToolCallDisplay {
  labelKey?: MessageKey;
  summary?: string;
}

function normalize(value: string): string | undefined {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized || undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? normalize(value) : undefined;
}

function objectValue(value: unknown): ArgumentsObject | undefined {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? value as ArgumentsObject
    : undefined;
}

function joinValues(...values: Array<string | undefined>): string | undefined {
  const present = values.filter((value): value is string => Boolean(value));
  return present.length > 0 ? present.join(" · ") : undefined;
}

const CONFIG: Record<string, ToolDisplayConfig> = {
  shell: { labelKey: "tool.name.shell", summary: (a) => stringValue(a.command) },
  code_interpreter: {
    labelKey: "tool.name.codeInterpreter",
    summary: (a) => joinValues(
      stringValue(a.language),
      typeof a.code === "string" ? normalize(a.code.split(/\r?\n/, 1)[0]) : undefined,
    ),
  },
  web_search: { labelKey: "tool.name.webSearch", summary: (a) => stringValue(a.query) },
  web_fetch: { labelKey: "tool.name.webFetch", summary: (a) => stringValue(a.url) },
  knowledge_search: { labelKey: "tool.name.knowledgeSearch", summary: (a) => stringValue(a.query) },
  knowledge_find: { labelKey: "tool.name.knowledgeFind", summary: (a) => stringValue(a.query) },
  knowledge_read: {
    labelKey: "tool.name.knowledgeRead",
    summary: (a) => stringValue(a.document_id) ?? stringValue(a.chunk_id),
  },
  knowledge_list: { labelKey: "tool.name.knowledgeList" },
  current_datetime: { labelKey: "tool.name.currentDatetime" },
  load_skill: { labelKey: "tool.name.loadSkill", summary: (a) => stringValue(a.skill_id) },
  enable_skill_for_agent: { labelKey: "tool.name.enableSkillForAgent", summary: (a) => stringValue(a.skill_id) },
  read_skill_resource: {
    labelKey: "tool.name.readSkillResource",
    summary: (a) => joinValues(stringValue(a.skill_id), stringValue(a.path)),
  },
  install_skill: {
    labelKey: "tool.name.installSkill",
    summary: (a) => {
      const source = objectValue(a.source);
      return source
        ? joinValues(stringValue(source.type), stringValue(source.url), stringValue(source.path), stringValue(source.upload_id))
        : undefined;
    },
  },
  publish_artifact: {
    labelKey: "tool.name.publishArtifact",
    summary: (a) => stringValue(a.name) ?? stringValue(a.path),
  },
  spawn_subagent: {
    labelKey: "tool.name.spawnSubagent",
    summary: (a) => joinValues(stringValue(a.agent_id), stringValue(a.task)),
  },
  read_handle: { labelKey: "tool.name.readHandle", summary: (a) => stringValue(a.handle) },
};

function parseArguments(rawArguments: string): ArgumentsObject | undefined {
  try {
    return objectValue(JSON.parse(rawArguments));
  } catch {
    return undefined;
  }
}

export function getToolCallDisplay(toolName: string, rawArguments: string): ToolCallDisplay {
  const config = CONFIG[toolName];
  if (!config) return {};

  const args = parseArguments(rawArguments);
  let summary: string | undefined;
  if (args && config.summary) {
    try {
      summary = config.summary(args);
    } catch {
      summary = undefined;
    }
  }
  return { labelKey: config.labelKey, summary };
}
```

- [ ] **Step 5: Run the focused test and verify GREEN**

Run:

```bash
cd frontend && npm test -- src/lib/__tests__/toolCallDisplay.test.ts
```

Expected: all display-policy tests PASS.

- [ ] **Step 6: Commit the display policy**

```bash
git add frontend/src/lib/toolCallDisplay.ts frontend/src/lib/__tests__/toolCallDisplay.test.ts frontend/src/i18n/en.ts frontend/src/i18n/zh.ts
git commit -m "feat(ui): define localized tool call display policy"
```

---

### Task 2: Tool Call Title Rendering

**Files:**
- Modify: `frontend/src/components/ToolCall.tsx`
- Modify: `frontend/src/components/__tests__/ToolCall.test.tsx`

**Interfaces:**
- Consumes: `getToolCallDisplay(tool.name, tool.arguments)` from Task 1.
- Produces: a localized, single-line tool-call trigger with a hoverable truncated summary and unchanged expanded content.

- [ ] **Step 1: Write failing component tests**

Extend `frontend/src/components/__tests__/ToolCall.test.tsx` with language setup and these cases:

```tsx
import { beforeEach, describe, it, expect } from "vitest";
import { useI18nStore } from "../../i18n";

beforeEach(() => {
  useI18nStore.getState().setLang("zh");
});

it("shows localized name, raw name, and hoverable summary", () => {
  render(<ToolCall tool={{ id: "c1", name: "shell", arguments: '{"command":"ls  /opt/code\\n--color=auto"}', status: "done", output: "" }} />);
  expect(screen.getByText("执行命令")).toBeInTheDocument();
  expect(screen.getByText("(shell)")).toBeInTheDocument();
  const summary = screen.getByTitle("ls /opt/code --color=auto");
  expect(summary).toHaveTextContent("ls /opt/code --color=auto");
  expect(summary).toHaveClass("truncate");
});

it("updates the localized tool name when language changes", () => {
  useI18nStore.getState().setLang("en");
  render(<ToolCall tool={{ id: "c1", name: "web_search", arguments: '{"query":"PAI-RAG"}', status: "running" }} />);
  expect(screen.getByText("Search the web")).toBeInTheDocument();
  expect(screen.getByText("(web_search)")).toBeInTheDocument();
});

it("shows a known no-argument tool without a summary separator", () => {
  render(<ToolCall tool={{ id: "c1", name: "current_datetime", arguments: "{}", status: "done", output: "now" }} />);
  expect(screen.getByText("获取当前时间")).toBeInTheDocument();
  expect(screen.getByText("(current_datetime)")).toBeInTheDocument();
  expect(screen.queryByText("·")).not.toBeInTheDocument();
});

it("keeps unknown tools as a raw name only", () => {
  render(<ToolCall tool={{ id: "c1", name: "custom_tool", arguments: '{"query":"x"}', status: "done", output: "ok" }} />);
  expect(screen.getByText("custom_tool")).toBeInTheDocument();
  expect(screen.queryByText("(custom_tool)")).not.toBeInTheDocument();
  expect(screen.queryByTitle("x")).not.toBeInTheDocument();
});
```

Update the two existing assertions from `getByText("web_fetch")` to `getByText("(web_fetch)")`, because built-in raw names now include parentheses.

- [ ] **Step 2: Run the focused component test and verify RED**

Run:

```bash
cd frontend && npm test -- src/components/__tests__/ToolCall.test.tsx
```

Expected: FAIL because `ToolCall` still renders only the raw tool name.

- [ ] **Step 3: Render the display policy with distinct visual levels**

In `frontend/src/components/ToolCall.tsx`, import the policy:

```ts
import { getToolCallDisplay } from "../lib/toolCallDisplay";
```

Inside `ToolCall`, derive the display value after `open`:

```ts
  const display = getToolCallDisplay(tool.name, tool.arguments);
```

Add `min-w-0` to the trigger class and replace the current raw-name span with:

```tsx
        {display.labelKey ? (
          <>
            <span className="shrink-0 text-xs font-semibold text-[var(--text)]">
              {t(display.labelKey)}
            </span>
            <span className="shrink-0 font-mono text-xs text-[var(--text-muted)]">
              ({tool.name})
            </span>
          </>
        ) : (
          <span className="shrink-0 font-mono text-xs font-medium text-[var(--text)]">
            {tool.name}
          </span>
        )}
        {display.summary && (
          <>
            <span aria-hidden="true" className="shrink-0 text-xs text-[var(--text-faint)]">·</span>
            <span
              className="min-w-0 truncate font-mono text-xs text-[var(--text-faint)]"
              title={display.summary}
            >
              {display.summary}
            </span>
          </>
        )}
```

Add `shrink-0` to the status, duration, and chevron elements so the summary is the only shrinking region. Keep status, duration, open state, expanded arguments, output, and error rendering otherwise unchanged.

- [ ] **Step 4: Run the focused component test and verify GREEN**

Run:

```bash
cd frontend && npm test -- src/components/__tests__/ToolCall.test.tsx
```

Expected: all `ToolCall` tests PASS.

- [ ] **Step 5: Run full frontend verification**

Run:

```bash
cd frontend && npm test && npm run build
```

Expected: complete frontend test suite PASS; TypeScript and Vite production build PASS with no errors.

- [ ] **Step 6: Commit the component integration**

```bash
git add frontend/src/components/ToolCall.tsx frontend/src/components/__tests__/ToolCall.test.tsx
git commit -m "feat(ui): show summaries in tool call titles"
```

---

### Task 3: Final Requirement Audit

**Files:**
- Verify: `frontend/src/lib/toolCallDisplay.ts`
- Verify: `frontend/src/components/ToolCall.tsx`
- Verify: `frontend/src/i18n/en.ts`
- Verify: `frontend/src/i18n/zh.ts`

**Interfaces:**
- Consumes: completed display policy and `ToolCall` integration.
- Produces: verification evidence that the implementation matches the approved design and leaves the worktree clean.

- [ ] **Step 1: Verify every backend built-in name has frontend policy**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

builtin = set()
for path in Path("backend/agent/tools/builtin").glob("*.py"):
    builtin.update(re.findall(r'name="([a-z_]+)"', path.read_text()))
policy = Path("frontend/src/lib/toolCallDisplay.ts").read_text()
missing = sorted(name for name in builtin if re.search(rf"^  {re.escape(name)}:", policy, re.M) is None)
assert not missing, f"missing built-in display policies: {missing}"
print(f"covered {len(builtin)} built-in tools")
PY
```

Expected: prints the covered count and exits successfully.

- [ ] **Step 2: Run formatting and repository-state checks**

Run:

```bash
git diff --check && git status --short
```

Expected: `git diff --check` exits successfully; `git status --short` is empty after the two implementation commits.

- [ ] **Step 3: Record the final verification result**

Report the exact focused-test, full-test, build, built-in coverage, and clean-worktree results. Do not create an empty audit commit when no files changed.
