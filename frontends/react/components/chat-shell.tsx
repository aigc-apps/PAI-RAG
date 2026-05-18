"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, Bot, Brain, CheckCircle2, ChevronDown, LoaderCircle, Plus, RefreshCw, Square, Trash2, User2, Wrench, X } from "lucide-react";
import {
  cancelSession,
  createSession,
  deleteSession,
  getActiveModel,
  getSession,
  getSkills,
  listSessions,
  setActiveModel,
  streamRegenerate,
  streamResponses,
} from "@/lib/api";
import type { AgentUpdate, AskUserPayload, ChatMessage, EvolvedSkill, OfficialSkill, PendingHitl, SessionSummary, SkillInventory } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";

type ProcessItem = {
  id: string;
  title: string;
  name?: string;
  kind: string;
  status: string;
  hidden?: boolean;
  input?: Record<string, unknown>;
  inputDraft?: string;
  output?: unknown;
  content?: string;
};

type ProcessGroup = {
  id: string;
  thought?: ProcessItem;
  items: ProcessItem[];
};

const INTERNAL_TOOL_NAMES = new Set(["update_working_checkpoint", "update_todo", "start_long_term_update", "final_report"]);
const MODEL_PROTOCOL_TAG_RE = /<\/?(?:summary|thinking|clinical[_-]thinking|checking|taking|taking[_-]action|skill[_-]context|working)\b[^>]*>/gi;
const STICKY_BOTTOM_PX = 96;
const RESUME_STICKY_PX = 640;

function escapeModelProtocolTags(content = "") {
  return content.replace(MODEL_PROTOCOL_TAG_RE, (tag) =>
    tag.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"),
  );
}

function Markdown({ content }: { content: string }) {
  const markdownContent = escapeModelProtocolTags(content);

  return (
    <div className="prose-agent">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdownContent}</ReactMarkdown>
    </div>
  );
}

function statusVariant(status: string) {
  if (status === "completed") {
    return "secondary" as const;
  }
  return "outline" as const;
}

function contentText(update: AgentUpdate) {
  if ("content" in update && update.content?.type === "text") {
    return update.content.text;
  }
  return "";
}

function isInternalToolName(name?: string) {
  return Boolean(name && INTERNAL_TOOL_NAMES.has(name));
}

function cleanInternalDisplayText(text = "") {
  return text
    .replace(/^\s*(?:\[Info\]\s*)?working memory updated\b.*$/gim, "")
    .replace(/^\s*\{?\s*["']result["']\s*:\s*["']working memory updated["']\s*\}?\s*$/gim, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function containsInternalDisplayText(value: unknown): boolean {
  if (typeof value === "string") {
    return /\bworking memory updated\b/i.test(value);
  }
  const record = asRecord(value);
  return typeof record?.result === "string" && record.result.toLowerCase() === "working memory updated";
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function parseJsonRecord(value?: string) {
  if (!value) {
    return null;
  }
  try {
    return asRecord(JSON.parse(value));
  } catch {
    return null;
  }
}

function stringField(record: Record<string, unknown> | null | undefined, key: string) {
  const value = record?.[key];
  return typeof value === "string" ? value : "";
}

function numberField(record: Record<string, unknown> | null | undefined, key: string) {
  const value = record?.[key];
  return typeof value === "number" ? value : null;
}

function isActiveStatus(status: string) {
  return status === "pending" || status === "in_progress";
}

function groupStatus(group: ProcessGroup) {
  const items = group.thought ? [group.thought, ...group.items] : group.items;
  if (items.some((item) => item.status === "failed")) {
    return "failed";
  }
  if (items.some((item) => isActiveStatus(item.status))) {
    return "in_progress";
  }
  return "completed";
}

function hasReasoningContent(item?: ProcessItem) {
  return Boolean(item?.content?.trim());
}

function groupProcessItems(items: ProcessItem[]) {
  const groups: ProcessGroup[] = [];
  let current: ProcessGroup | null = null;

  for (const item of items) {
    if (item.kind === "think") {
      current = {
        id: `agent-step-${item.id}`,
        thought: item,
        items: [],
      };
      groups.push(current);
      continue;
    }

    if (!current) {
      current = {
        id: `agent-step-${item.id}`,
        items: [],
      };
      groups.push(current);
    }

    current.items.push(item);
  }

  return groups.filter((group) => group.items.length > 0 || hasReasoningContent(group.thought));
}

function buildProcessItems(events: AgentUpdate[] = []) {
  const items: ProcessItem[] = [];
  const byId = new Map<string, ProcessItem>();

  for (const event of events) {
    if (event.sessionUpdate === "thought_start") {
      const item: ProcessItem = {
        id: event.thoughtId,
        title: event.title || "Agent step",
        kind: "think",
        status: event.status || "in_progress",
        hidden: event.hidden,
        content: contentText(event),
      };
      byId.set(event.thoughtId, item);
      items.push(item);
      continue;
    }

    if (event.sessionUpdate === "thought_delta") {
      let item = byId.get(event.thoughtId);
      if (!item) {
        item = {
          id: event.thoughtId,
          title: "Agent step",
          kind: "think",
          status: "in_progress",
        };
        byId.set(event.thoughtId, item);
        items.push(item);
      }
      const text = contentText(event);
      if (text) {
        item.content = event.replace ? text : `${item.content || ""}${text}`;
      }
      continue;
    }

    if (event.sessionUpdate === "thought_done") {
      let item = byId.get(event.thoughtId);
      if (!item) {
        item = {
          id: event.thoughtId,
          title: "Agent step",
          kind: "think",
          status: event.status,
        };
        byId.set(event.thoughtId, item);
        items.push(item);
      }
      item.status = event.status;
      item.hidden = event.hidden ?? item.hidden;
      if (event.content) {
        item.content = contentText(event);
      }
      continue;
    }

    if (event.sessionUpdate === "thought") {
      items.push({
        id: `thought-${items.length}`,
        title: event.title || "Thinking",
        kind: "think",
        status: "completed",
        content: contentText(event),
      });
      continue;
    }

    if (event.sessionUpdate === "tool_call_delta") {
      const name = event.name || "";
      let item = byId.get(event.toolCallId);
      if (!item) {
        item = {
          id: event.toolCallId,
          title: event.title || name || "Tool call",
          name,
          kind: event.kind || "tool",
          status: event.status || "in_progress",
          hidden: event.hidden || isInternalToolName(name),
          inputDraft: event.argumentsText || event.argumentsDelta || "",
        };
        byId.set(event.toolCallId, item);
        items.push(item);
      } else {
        if (event.title) {
          item.title = event.title;
        }
        if (name) {
          item.name = name;
        }
        if (event.kind) {
          item.kind = event.kind;
        }
        item.status = event.status || item.status;
        item.hidden = event.hidden ?? item.hidden;
        if (isInternalToolName(item.name)) {
          item.hidden = true;
        }
        if (typeof event.argumentsText === "string") {
          item.inputDraft = event.argumentsText;
        } else if (event.argumentsDelta) {
          item.inputDraft = `${item.inputDraft || ""}${event.argumentsDelta}`;
        }
      }
      continue;
    }

    if (event.sessionUpdate === "tool_call") {
      let item = byId.get(event.toolCallId);
      if (!item) {
        item = {
          id: event.toolCallId,
          title: event.title || event.name,
          name: event.name,
          kind: event.kind || "tool",
          status: event.status || "pending",
          hidden: event.hidden || isInternalToolName(event.name),
        };
        byId.set(event.toolCallId, item);
        items.push(item);
      }
      item.title = event.title || item.title || event.name;
      item.name = event.name || item.name;
      item.kind = event.kind || item.kind || "tool";
      item.status = event.status || item.status || "pending";
      item.hidden = event.hidden || item.hidden || isInternalToolName(event.name);
      item.input = event.input;
      item.inputDraft = undefined;
      continue;
    }

    if (event.sessionUpdate === "tool_call_update") {
      let item = byId.get(event.toolCallId);
      if (!item) {
        item = {
          id: event.toolCallId,
          title: "Tool call",
          kind: "tool",
          status: event.status,
          hidden: containsInternalDisplayText(event.data) || containsInternalDisplayText(contentText(event)),
        };
        byId.set(event.toolCallId, item);
        items.push(item);
      }
      if (containsInternalDisplayText(event.data) || containsInternalDisplayText(contentText(event))) {
        item.hidden = true;
      }
      item.status = event.status;
      item.output = event.data ?? item.output;
      const text = cleanInternalDisplayText(contentText(event));
      if (text) {
        item.content = item.content ? `${item.content}\n\n${text}` : text;
      }
    }
  }

  return items.filter(
    (item) =>
      !item.hidden &&
      (item.content ||
        item.input ||
        item.inputDraft ||
        item.output ||
        item.kind === "think" ||
        item.status === "pending" ||
        item.status === "in_progress"),
  );
}

function askEvents(events: AgentUpdate[] = []): AskUserPayload[] {
  return events
    .filter((event): event is Extract<AgentUpdate, { sessionUpdate: "ask_user" }> => event.sessionUpdate === "ask_user")
    .map((event) => ({ question: event.question, candidates: event.candidates }));
}

function latestAsk(messages: ChatMessage[]) {
  const last = messages[messages.length - 1];
  if (last?.role !== "assistant") {
    return null;
  }
  const asks = askEvents(last.events);
  return asks.length ? asks[asks.length - 1] : null;
}

function fallbackDisplayContent(events: AgentUpdate[] = []) {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.sessionUpdate === "thought_done" && event.status === "completed" && !event.hidden) {
      const text = cleanInternalDisplayText(event.content?.text || "");
      if (text) {
        return text;
      }
    }
  }
  return "";
}

function mergeDetailMessages(detail: { messages?: ChatMessage[] | null; pending_hitl?: PendingHitl | null }): ChatMessage[] {
  const base = detail.messages ?? [];
  const pending = detail.pending_hitl ?? null;
  if (!pending || pending.tool_name !== "ask_user" || !pending.question) {
    return base;
  }
  const last = base[base.length - 1];
  const lastIsAsk =
    last?.role === "assistant" &&
    (last.events ?? []).some(
      (event) => event.sessionUpdate === "ask_user" && event.question === pending.question,
    );
  if (lastIsAsk) {
    return base;
  }
  const askEvent: AgentUpdate = {
    sessionUpdate: "ask_user",
    question: pending.question,
    candidates: pending.candidates,
  };
  if (last?.role === "assistant") {
    return [
      ...base.slice(0, -1),
      { ...last, events: [...(last.events ?? []), askEvent] },
    ];
  }
  return [...base, { role: "assistant", content: "", events: [askEvent] }];
}

function sameUserTurn(a: ChatMessage | undefined, b: ChatMessage | undefined) {
  return a?.role === "user" && b?.role === "user" && a.content === b.content;
}

function lastUserIndex(messages: ChatMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") {
      return index;
    }
  }
  return -1;
}

// After a freshly-finished stream, the in-memory transcript holds a richer
// view than the server snapshot — it carries reasoning_step / tool_call
// events that the backend doesn't persist into ui_messages. Keep the last
// in-memory assistant message (with its events) when the server returns the
// same number of trailing assistant messages, only refreshing earlier turns
// from the server. If the server snapshot is stale and doesn't include the
// just-submitted user turn yet, keep the local transcript so the answer
// doesn't flash and disappear. Falls back to mergeDetailMessages when prev
// has nothing.
function mergePostStreamMessages(
  prev: ChatMessage[],
  detail: { messages?: ChatMessage[] | null; pending_hitl?: PendingHitl | null },
): ChatMessage[] {
  const merged = mergeDetailMessages(detail);
  const prevLast = prev[prev.length - 1];
  const prevUser = prev[prev.length - 2];
  if (prevLast?.role === "assistant" && prevUser?.role === "user") {
    const mergedLastUser = lastUserIndex(merged);
    if (mergedLastUser === -1 || !sameUserTurn(merged[mergedLastUser], prevUser)) {
      return prev;
    }
    if (mergedLastUser === merged.length - 1) {
      return [...merged, prevLast];
    }
  }
  if (!prevLast || prevLast.role !== "assistant") {
    return merged;
  }
  const prevEventCount = prevLast.events?.length ?? 0;
  const mergedLast = merged[merged.length - 1];
  const mergedEventCount = mergedLast?.events?.length ?? 0;
  const prevContentLen = (prevLast.content || "").length;
  const mergedContentLen = (mergedLast?.content || "").length;
  if (
    mergedLast?.role === "assistant" &&
    prevEventCount >= mergedEventCount &&
    prevContentLen >= mergedContentLen
  ) {
    return [...merged.slice(0, -1), prevLast];
  }
  if (mergedLast?.role !== "assistant" && prevEventCount > 0) {
    return [...merged, prevLast];
  }
  return merged;
}

function applyAssistantUpdate(messages: ChatMessage[], update: AgentUpdate): ChatMessage[] {
  const next = [...messages];
  let last = next[next.length - 1];
  if (!last || last.role !== "assistant") {
    last = { role: "assistant", content: "", events: [] };
    next.push(last);
  }

  if (update.sessionUpdate === "agent_message_chunk") {
    next[next.length - 1] = {
      ...last,
      content: `${last.content || ""}${update.content.text || ""}`,
      events: last.events ?? [],
    };
    return next;
  }

  next[next.length - 1] = {
    ...last,
    events: [...(last.events ?? []), update],
  };
  return next;
}

function AskCard({
  ask,
  disabled,
  onSelectCandidate,
}: {
  ask: AskUserPayload;
  disabled?: boolean;
  onSelectCandidate?: (candidate: string, index: number) => void;
}) {
  return (
    <Card className="border-blue-100 bg-blue-50/70 shadow-none">
      <CardContent className="space-y-3 p-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-normal text-blue-700">Agent asks</span>
          <Badge variant="secondary">Input needed</Badge>
        </div>
        <Markdown content={ask.question || "Please provide input."} />
        {ask.candidates?.length ? (
          <div className="space-y-2">
            {ask.candidates.map((candidate, index) => (
              <button
                className="flex w-full gap-2.5 rounded-md border border-blue-100 bg-white/80 p-2.5 text-left transition-colors hover:border-blue-300 hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={disabled}
                key={`${candidate}-${index}`}
                onClick={() => onSelectCandidate?.(candidate, index)}
                type="button"
              >
                <span className="mt-0.5 text-sm font-semibold text-blue-700">{index + 1}</span>
                <span className="min-w-0 whitespace-pre-wrap text-sm leading-relaxed text-slate-700">{candidate}</span>
              </button>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function CodeRunDetails({ item }: { item: ProcessItem }) {
  const input = asRecord(item.input);
  const output = asRecord(item.output) ?? parseJsonRecord(item.content);
  const codeType = stringField(input, "type") || "python";
  const script = stringField(input, "script");
  const cwd = stringField(input, "cwd");
  const timeout = numberField(input, "timeout");
  const inputDraft = item.inputDraft?.trim() || "";
  const status = stringField(output, "status") || item.status;
  const exitCode = numberField(output, "exit_code");
  const stdout = stringField(output, "stdout") || item.content || "";
  const msg = stringField(output, "msg");

  return (
    <div className="space-y-2">
      {script ? (
        <details className="overflow-hidden rounded-md border border-slate-200 bg-slate-50" open>
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-2.5 py-1.5 text-sm font-medium text-slate-700">
            <span>Script</span>
            <span className="flex shrink-0 flex-wrap items-center justify-end gap-2 text-xs font-normal">
              <Badge variant="outline">{codeType}</Badge>
              {cwd ? <Badge variant="outline">{cwd}</Badge> : null}
              {timeout !== null ? <Badge variant="outline">{timeout}s</Badge> : null}
            </span>
          </summary>
          <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-words border-t border-slate-200 bg-slate-950 p-2.5 text-xs leading-relaxed text-slate-50">
            {script}
          </pre>
        </details>
      ) : null}
      {!script && inputDraft ? (
        <details className="overflow-hidden rounded-md border border-slate-200 bg-slate-50" open>
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-2.5 py-1.5 text-sm font-medium text-slate-700">
            <span>Input draft</span>
            <Badge variant="outline">{codeType}</Badge>
          </summary>
          <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-words border-t border-slate-200 bg-slate-950 p-2.5 text-xs leading-relaxed text-slate-50">
            {inputDraft}
          </pre>
        </details>
      ) : null}

      <div className="overflow-hidden rounded-md border border-slate-200 bg-white">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 px-2.5 py-1.5">
          <span className="text-sm font-medium text-slate-700">Result</span>
          <span className="flex items-center gap-2">
            <Badge
              className={status === "error" || item.status === "failed" ? "border-red-200 bg-red-50 text-red-700" : undefined}
              variant={status === "success" || item.status === "completed" ? "secondary" : "outline"}
            >
              {status}
            </Badge>
            {exitCode !== null ? <Badge variant="outline">exit {exitCode}</Badge> : null}
          </span>
        </div>
        {stdout ? (
          <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words p-2.5 font-mono text-xs leading-relaxed text-slate-800">
            {stdout}
          </pre>
        ) : (
          <div className="px-2.5 py-2 text-sm text-slate-500">
            {item.status === "in_progress" ? "Waiting for output..." : "No stdout output"}
          </div>
        )}
        {msg ? <div className="border-t border-slate-200 px-2.5 py-1.5 text-sm text-red-700">{msg}</div> : null}
      </div>
    </div>
  );
}

function GenericToolDetails({ item }: { item: ProcessItem }) {
  const hasBody = Boolean(item.input || item.inputDraft || item.output || item.content);

  return (
    <div className="space-y-2 rounded-md border border-slate-200 bg-white px-2.5 py-2">
      {item.input ? (
        <details className="overflow-hidden rounded border border-slate-200 bg-slate-50" open>
          <summary className="cursor-pointer list-none px-2.5 py-1.5 text-sm font-medium text-slate-700">Input</summary>
          <pre className="max-h-44 overflow-auto whitespace-pre-wrap break-words border-t border-slate-200 bg-slate-950 p-2.5 text-xs text-slate-50">
            {JSON.stringify(item.input, null, 2)}
          </pre>
        </details>
      ) : null}
      {!item.input && item.inputDraft ? (
        <details className="overflow-hidden rounded border border-slate-200 bg-slate-50" open>
          <summary className="cursor-pointer list-none px-2.5 py-1.5 text-sm font-medium text-slate-700">Input draft</summary>
          <pre className="max-h-44 overflow-auto whitespace-pre-wrap break-words border-t border-slate-200 bg-slate-950 p-2.5 text-xs text-slate-50">
            {item.inputDraft}
          </pre>
        </details>
      ) : null}
      {item.output && !item.content ? (
        <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-words rounded bg-slate-50 p-2.5 text-xs text-slate-700">
          {JSON.stringify(item.output, null, 2)}
        </pre>
      ) : null}
      {item.content ? (
        <div className="max-h-72 overflow-auto rounded bg-slate-50 p-2.5">
          <Markdown content={item.content} />
        </div>
      ) : null}
      {!hasBody ? (
        <div className="rounded bg-slate-50 px-2.5 py-2 text-sm text-slate-500">
          {item.status === "in_progress" || item.status === "pending" ? "Waiting for output..." : "No additional output"}
        </div>
      ) : null}
    </div>
  );
}

function itemTitle(item: ProcessItem) {
  if (item.kind === "think" && (item.title === "Agent step" || item.title === "Thinking")) {
    return "Reasoning";
  }
  return item.title;
}

function innerStepNumberClass(status: string) {
  if (status === "failed") {
    return "border-red-200 bg-red-50 text-red-700";
  }
  if (isActiveStatus(status)) {
    return "border-emerald-200 bg-emerald-600 text-white";
  }
  return "border-emerald-200 bg-emerald-50 text-emerald-700";
}

function AgentReasoningContent({ item }: { item?: ProcessItem }) {
  if (!item) {
    return null;
  }

  const content = cleanInternalDisplayText(item.content);
  if (!content) {
    return null;
  }

  const active = isActiveStatus(item.status);

  return (
    <div
      className={cn(
        "rounded-md border px-3 py-2 text-sm leading-relaxed",
        active ? "border-blue-200 bg-blue-50/50" : "border-slate-200 bg-slate-50/80",
      )}
    >
      <Markdown content={content} />
    </div>
  );
}

function ThinkDetails({ item }: { item: ProcessItem }) {
  const content = cleanInternalDisplayText(item.content);
  if (!content) {
    return (
      <div className="rounded bg-slate-50 px-2.5 py-2 text-sm text-slate-500">
        {item.status === "in_progress" || item.status === "pending"
          ? "Thinking..."
          : "No reasoning content"}
      </div>
    );
  }
  return (
    <div className="max-h-72 overflow-auto rounded bg-slate-50 p-2.5 text-sm leading-relaxed">
      <Markdown content={content} />
    </div>
  );
}

function ProcessSubStep({ item, stepNumber }: { item: ProcessItem; stepNumber: number }) {
  const isThought = item.kind === "think";
  const active = isActiveStatus(item.status);
  const [open, setOpen] = useState(active || isThought);

  useEffect(() => {
    if (active) {
      setOpen(true);
    }
  }, [active]);

  return (
    <details
      className={cn(
        "group/substep rounded-md border bg-white transition-colors",
        active ? "border-blue-200 bg-blue-50/30" : "border-slate-200",
      )}
      onToggle={(event) => setOpen(event.currentTarget.open)}
      open={open}
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-1.5">
        <span className="flex min-w-0 items-center gap-2">
          <span
            className={cn(
              "grid h-5 w-5 shrink-0 place-items-center rounded-full border text-xs font-semibold",
              innerStepNumberClass(item.status),
            )}
          >
            {stepNumber}
          </span>
          <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full border border-emerald-100 bg-white text-emerald-600">
            {isThought ? <Brain className="h-3 w-3" /> : <Wrench className="h-3 w-3" />}
          </span>
          <span className="truncate text-sm font-medium text-slate-700">{itemTitle(item)}</span>
        </span>
        <span className="flex shrink-0 items-center gap-2">
          <Badge
            className={cn(
              item.status === "failed" ? "border-red-200 bg-red-50 text-red-700" : undefined,
              active ? "border-blue-200 bg-blue-50 text-blue-700" : undefined,
            )}
            variant={statusVariant(item.status)}
          >
            {active ? "streaming" : item.status}
          </Badge>
          {!isThought ? <Badge variant="outline">{item.kind}</Badge> : null}
          <ChevronDown className="h-4 w-4 text-slate-400 transition-transform group-open/substep:rotate-180" />
        </span>
      </summary>
      <div className="border-t border-slate-200 p-2.5">
        {item.kind === "execute" ? (
          <CodeRunDetails item={item} />
        ) : item.kind === "think" ? (
          <ThinkDetails item={item} />
        ) : (
          <GenericToolDetails item={item} />
        )}
      </div>
    </details>
  );
}

function ProcessGroupCard({
  group,
  groupNumber,
  currentActiveGroupNumber,
}: {
  group: ProcessGroup;
  groupNumber: number;
  currentActiveGroupNumber: number;
}) {
  const status = groupStatus(group);
  const active = isActiveStatus(status);
  const [open, setOpen] = useState(active);

  useEffect(() => {
    if (!currentActiveGroupNumber) {
      return;
    }
    if (groupNumber === currentActiveGroupNumber) {
      setOpen(true);
    } else if (groupNumber < currentActiveGroupNumber) {
      setOpen(false);
    }
  }, [currentActiveGroupNumber, groupNumber]);

  return (
    <details
      className={cn(
        "group/agent overflow-hidden rounded-lg border bg-white transition-colors",
        active ? "border-blue-200 bg-blue-50/30" : "border-slate-200",
      )}
      onToggle={(event) => setOpen(event.currentTarget.open)}
      open={open}
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-2.5">
        <span className="flex min-w-0 items-center gap-3">
          <span
            className={cn(
              "grid h-8 w-8 shrink-0 place-items-center rounded-full border text-sm font-semibold",
              status === "failed"
                ? "border-red-200 bg-red-50 text-red-700"
                : active
                  ? "border-blue-200 bg-blue-600 text-white"
                  : "border-blue-200 bg-blue-50 text-blue-700",
            )}
          >
            {groupNumber}
          </span>
          <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full border border-slate-200 bg-white text-slate-400">
            <Brain className="h-3.5 w-3.5" />
          </span>
          <span className="truncate text-sm font-semibold text-slate-800">Agent step {groupNumber}</span>
        </span>
        <span className="flex shrink-0 items-center gap-2">
          <Badge
            className={cn(
              status === "failed" ? "border-red-200 bg-red-50 text-red-700" : undefined,
              active ? "border-blue-200 bg-blue-50 text-blue-700" : undefined,
            )}
            variant={statusVariant(status)}
          >
            {active ? "streaming" : status}
          </Badge>
          <Badge variant="outline">
            {group.items.length} {group.items.length === 1 ? "action" : "actions"}
          </Badge>
          <ChevronDown className="h-4 w-4 text-slate-400 transition-transform group-open/agent:rotate-180" />
        </span>
      </summary>
      <div className="space-y-2.5 border-t border-slate-200 p-3">
        {hasReasoningContent(group.thought) ? (
          <ProcessSubStep item={group.thought!} key={group.thought!.id} stepNumber={1} />
        ) : null}
        {group.items.map((item, index) => (
          <ProcessSubStep
            item={item}
            key={item.id}
            stepNumber={index + 1 + (hasReasoningContent(group.thought) ? 1 : 0)}
          />
        ))}
      </div>
    </details>
  );
}

function ProcessBlock({ items, streaming }: { items: ProcessItem[]; streaming: boolean }) {
  const groups = useMemo(() => groupProcessItems(items), [items]);
  const active = groups.some((group) => isActiveStatus(groupStatus(group)));
  const currentActiveGroupNumber = groups.reduce(
    (latest, group, index) => (isActiveStatus(groupStatus(group)) ? index + 1 : latest),
    0,
  );
  const title = active ? "Working through agent steps" : "Agent steps";
  const totalItems = groups.reduce(
    (total, group) =>
      total + group.items.length + (hasReasoningContent(group.thought) ? 1 : 0),
    0,
  );

  if (!groups.length) {
    return null;
  }

  return (
    <details className="group overflow-hidden rounded-lg border border-slate-200 bg-slate-50/70" open={streaming || active}>
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-2.5 text-sm text-slate-600">
        <span className="flex min-w-0 items-center gap-3">
          <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full border border-slate-200 bg-white text-slate-500">
            {streaming || active ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
          </span>
          <span className="truncate font-medium">{title}</span>
        </span>
        <span className="flex shrink-0 items-center gap-2 text-xs text-slate-400">
          {groups.length} agent steps / {totalItems} actions
          <ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" />
        </span>
      </summary>
      <div className="border-t border-slate-200 px-3 py-3">
        <div className="space-y-2.5">
          {groups.map((group, index) => (
            <ProcessGroupCard
              currentActiveGroupNumber={currentActiveGroupNumber}
              group={group}
              groupNumber={index + 1}
              key={group.id}
            />
          ))}
        </div>
      </div>
    </details>
  );
}

function MessageContent({
  message,
  onSelectCandidate,
  streaming = false,
}: {
  message: ChatMessage;
  onSelectCandidate?: (candidate: string, index: number) => void;
  streaming?: boolean;
}) {
  const events = message.events ?? [];
  const processItems = useMemo(() => buildProcessItems(events), [events]);
  const asks = useMemo(() => askEvents(events), [events]);
  const displayContent = useMemo(() => cleanInternalDisplayText(message.content), [message.content]);
  const hasProcessItems = processItems.length > 0;
  const hasAsks = asks.length > 0;
  const hasContent = Boolean(displayContent);
  const showContent = hasContent;
  const shouldUseFallbackContent = !streaming && !hasContent && !hasProcessItems && !hasAsks;
  const fallbackContent = useMemo(
    () => (shouldUseFallbackContent ? fallbackDisplayContent(events) : ""),
    [shouldUseFallbackContent, events],
  );
  const hasFallbackContent = Boolean(fallbackContent);
  const hasVisibleContent = showContent || hasFallbackContent || hasProcessItems || hasAsks;

  if (!hasVisibleContent) {
    return streaming ? (
      <span className="inline-flex h-4 w-1 animate-pulse rounded bg-blue-500" />
    ) : (
      <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
        No displayable response was returned.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {hasProcessItems ? <ProcessBlock items={processItems} streaming={streaming} /> : null}
      {showContent ? <Markdown content={displayContent} /> : null}
      {!showContent && hasFallbackContent ? <Markdown content={fallbackContent} /> : null}
      {asks.map((ask, index) => (
        <AskCard ask={ask} disabled={streaming} key={`${ask.question}-${index}`} onSelectCandidate={onSelectCandidate} />
      ))}
      {streaming ? <span className="mt-1 inline-flex h-4 w-1 animate-pulse rounded bg-blue-500" /> : null}
    </div>
  );
}

function sessionTitle(session: SessionSummary) {
  return session.title?.trim() || "New Task";
}

function titleFromMessages(messages: ChatMessage[]) {
  return messages.find((message) => message.role === "user" && message.content.trim())?.content.trim().slice(0, 60) || "New Task";
}

function canRegenerateLastAssistant(messages: ChatMessage[]) {
  const last = messages[messages.length - 1];
  const previous = messages[messages.length - 2];
  return Boolean(
    last?.role === "assistant" &&
      previous?.role === "user" &&
      previous.content.trim() &&
      (last.content.trim() || (last.events?.length ?? 0) > 0),
  );
}

function OfficialSkillRow({ skill }: { skill: OfficialSkill }) {
  return (
    <div className="rounded-md border border-blue-200 bg-blue-50/50 px-3 py-2.5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-blue-950">{skill.name}</div>
          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-blue-800/70">{skill.description || "No description"}</p>
        </div>
        <Badge className="max-w-48 shrink-0 truncate border-blue-200 bg-blue-100 text-blue-700" variant="outline">{skill.trigger}</Badge>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-blue-700/70">
        <span className="min-w-0 truncate font-mono">{skill.source}</span>
        {skill.allowed_tools.length ? <Badge className="border-blue-200 text-blue-700" variant="outline">{skill.allowed_tools.length} tools</Badge> : null}
      </div>
    </div>
  );
}

function EvolvedSkillRow({ skill }: { skill: EvolvedSkill }) {
  return (
    <div className="rounded-md border border-emerald-200 bg-emerald-50/50 px-3 py-2.5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-emerald-950">{skill.name}</div>
          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-emerald-800/70">{skill.description || "No description"}</p>
        </div>
        <Badge className="border-emerald-200 bg-emerald-100 text-emerald-700" variant="outline">{skill.kind}</Badge>
      </div>
      <div className="mt-2 min-w-0 truncate font-mono text-xs text-emerald-700/70">{skill.source}</div>
    </div>
  );
}

function SkillsDialog({
  inventory,
  loading,
  error,
  onClose,
}: {
  inventory: SkillInventory | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}) {
  const official = inventory?.official ?? [];
  const evolved = inventory?.evolved ?? [];

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-900/30 px-4 py-6 backdrop-blur-sm" role="dialog" aria-modal="true">
      <div className="flex max-h-[82vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border border-slate-200 bg-white shadow-xl">
        <div className="flex items-center justify-between gap-4 border-b border-slate-200 px-5 py-4">
          <div>
            <h3 className="text-base font-semibold text-slate-900">Skills</h3>
            <p className="text-sm text-slate-500">
              {official.length} official / {evolved.length} self-evolved
            </p>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close skills">
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {loading ? (
            <div className="flex items-center gap-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-3 text-sm text-slate-600">
              <LoaderCircle className="h-4 w-4 animate-spin" />
              Loading skills...
            </div>
          ) : null}
          {error ? (
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-3 text-sm text-red-700">{error}</div>
          ) : null}
          {!loading && !error ? (
            <div className="space-y-5">
              <section>
                <div className="mb-2 flex items-center justify-between gap-3">
                  <h4 className="text-sm font-semibold text-blue-800">Official</h4>
                  <Badge className="bg-blue-100 text-blue-700" variant="secondary">{official.length}</Badge>
                </div>
                <div className="space-y-2">
                  {official.length ? official.map((skill) => <OfficialSkillRow key={skill.source} skill={skill} />) : (
                    <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-3 text-sm text-slate-500">
                      No official skills configured.
                    </div>
                  )}
                </div>
              </section>
              <section>
                <div className="mb-2 flex items-center justify-between gap-3">
                  <h4 className="text-sm font-semibold text-emerald-800">Self-evolved</h4>
                  <Badge className="bg-emerald-100 text-emerald-700" variant="secondary">{evolved.length}</Badge>
                </div>
                <div className="space-y-2">
                  {evolved.length ? evolved.map((skill) => <EvolvedSkillRow key={skill.source} skill={skill} />) : (
                    <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-3 text-sm text-slate-500">
                      No self-evolved skills yet.
                    </div>
                  )}
                </div>
              </section>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function shouldReplaceSessionTitle(title?: string) {
  const value = title?.trim();
  return !value || value === "New Task";
}

const ACTIVE_MODEL_LS_KEY = "pai-rag.active-model";

function ModelPickerDialog({
  current,
  onClose,
  onSaved,
}: {
  current: string;
  onClose: () => void;
  onSaved: (name: string) => void;
}) {
  const [value, setValue] = useState(current);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const submit = async () => {
    const trimmed = value.trim();
    if (!trimmed || busy) {
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const res = await setActiveModel(trimmed);
      onSaved(res.active_model);
      onClose();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "保存失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-900/30 px-4 py-6 backdrop-blur-sm" role="dialog" aria-modal="true">
      <div className="flex w-full max-w-md flex-col overflow-hidden rounded-lg border border-slate-200 bg-white shadow-xl">
        <div className="flex items-center justify-between gap-4 border-b border-slate-200 px-5 py-4">
          <div>
            <h3 className="text-base font-semibold text-slate-900">切换模型</h3>
            <p className="text-sm text-slate-500">输入要切换到的模型名,例如 qwen-plus、qwen-max、glm-5</p>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close">
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="space-y-3 px-5 py-4">
          <input
            autoFocus
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void submit();
              } else if (e.key === "Escape") {
                e.preventDefault();
                onClose();
              }
            }}
            placeholder="qwen-plus"
            className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />
          {err ? (
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{err}</div>
          ) : null}
          <p className="text-xs text-slate-500">
            模型名不会预先校验,如果上游不识别会在下次对话时报错。修改后立即对所有新对话生效,正在进行中的对话沿用原模型。
          </p>
        </div>
        <div className="flex items-center justify-end gap-2 border-t border-slate-200 px-5 py-3">
          <Button variant="ghost" onClick={onClose} disabled={busy}>
            取消
          </Button>
          <Button onClick={() => void submit()} disabled={busy || !value.trim()}>
            {busy ? "保存中…" : "保存"}
          </Button>
        </div>
      </div>
    </div>
  );
}

export function ChatShell() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messagesBySession, setMessagesBySession] = useState<Record<string, ChatMessage[]>>({});
  const [streamingSessions, setStreamingSessions] = useState<Set<string>>(() => new Set());
  const [pendingHitlBySession, setPendingHitlBySession] = useState<Record<string, PendingHitl | null>>({});
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [skillsOpen, setSkillsOpen] = useState(false);
  const [skillsInventory, setSkillsInventory] = useState<SkillInventory | null>(null);
  const [skillsLoading, setSkillsLoading] = useState(false);
  const [skillsError, setSkillsError] = useState<string | null>(null);
  const [activeModel, setActiveModelState] = useState<string>(() => {
    if (typeof window === "undefined") return "";
    try {
      return localStorage.getItem(ACTIVE_MODEL_LS_KEY) ?? "";
    } catch {
      return "";
    }
  });
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const abortControllers = useRef<Map<string, AbortController>>(new Map());
  const streamingSessionsRef = useRef<Set<string>>(streamingSessions);
  const messagesViewportRef = useRef<HTMLDivElement | null>(null);
  const shouldStickToBottomRef = useRef(true);
  const programmaticScrollRef = useRef(false);
  const lastScrollTopRef = useRef(0);
  const lastTouchYRef = useRef<number | null>(null);

  streamingSessionsRef.current = streamingSessions;

  const messages = currentSessionId ? messagesBySession[currentSessionId] ?? [] : [];
  const streaming = currentSessionId ? streamingSessions.has(currentSessionId) : false;
  const currentPendingHitl = currentSessionId ? pendingHitlBySession[currentSessionId] ?? null : null;

  const setSessionMessages = useCallback(
    (sessionId: string, updater: (prev: ChatMessage[]) => ChatMessage[]) => {
      setMessagesBySession((prev) => ({ ...prev, [sessionId]: updater(prev[sessionId] ?? []) }));
    },
    [],
  );

  const markStreaming = useCallback((sessionId: string, on: boolean) => {
    setStreamingSessions((prev) => {
      const has = prev.has(sessionId);
      if (on === has) {
        return prev;
      }
      const next = new Set(prev);
      if (on) {
        next.add(sessionId);
      } else {
        next.delete(sessionId);
      }
      return next;
    });
  }, []);

  const setSessionPendingHitl = useCallback((sessionId: string, pending: PendingHitl | null) => {
    setPendingHitlBySession((prev) => {
      if (!pending) {
        if (!(sessionId in prev) || prev[sessionId] === null) {
          if (prev[sessionId] === null) {
            return prev;
          }
        }
        if (!(sessionId in prev)) {
          return prev;
        }
        const next = { ...prev };
        delete next[sessionId];
        return next;
      }
      if (prev[sessionId] && prev[sessionId]?.call_id === pending.call_id && prev[sessionId]?.response_id === pending.response_id) {
        return prev;
      }
      return { ...prev, [sessionId]: pending };
    });
  }, []);

  const refreshSessions = useCallback(async () => {
    const data = await listSessions();
    setSessions(data);
    return data;
  }, []);

  const refreshSkills = useCallback(async () => {
    setSkillsLoading(true);
    setSkillsError(null);
    try {
      setSkillsInventory(await getSkills());
    } catch (err) {
      setSkillsError(err instanceof Error ? err.message : String(err));
    } finally {
      setSkillsLoading(false);
    }
  }, []);

  const loadSession = useCallback(
    async (sessionId: string) => {
      shouldStickToBottomRef.current = true;
      setShowScrollToBottom(false);
      setCurrentSessionId(sessionId);
      // Streaming sessions hold a fresher in-memory transcript than the
      // server snapshot (which is debounced up to ~750ms behind). Don't
      // clobber it; just refresh the sidebar.
      if (streamingSessionsRef.current.has(sessionId)) {
        await refreshSessions();
        return;
      }
      const detail = await getSession(sessionId);
      setSessionMessages(detail.session_id, () => mergeDetailMessages(detail));
      setSessionPendingHitl(detail.session_id, detail.pending_hitl ?? null);
      if (detail.session_id !== sessionId) {
        setCurrentSessionId(detail.session_id);
      }
      await refreshSessions();
    },
    [refreshSessions, setSessionMessages, setSessionPendingHitl],
  );

  const loadInitialSession = useCallback(async () => {
    const data = await refreshSessions();
    if (data.length) {
      await loadSession(data[0].session_id);
    } else {
      const created = await createSession();
      shouldStickToBottomRef.current = true;
      setShowScrollToBottom(false);
      setCurrentSessionId(created.session_id);
      setSessionMessages(created.session_id, () => created.messages ?? []);
      await refreshSessions();
    }
  }, [loadSession, refreshSessions, setSessionMessages]);

  useEffect(() => {
    let active = true;
    async function boot() {
      try {
        setError(null);
        await loadInitialSession();
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }
    void boot();
    return () => {
      active = false;
    };
  }, [loadInitialSession]);

  useEffect(() => {
    let cancelled = false;
    void getActiveModel()
      .then((res) => {
        if (cancelled) return;
        setActiveModelState(res.active_model);
        try {
          localStorage.setItem(ACTIVE_MODEL_LS_KEY, res.active_model);
        } catch {
          // ignore quota / private-mode issues
        }
      })
      .catch(() => {
        // best-effort: keep showing whatever's in state (localStorage cache or empty)
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const scrollViewportToBottom = useCallback((viewport: HTMLDivElement, behavior: ScrollBehavior = "auto") => {
    programmaticScrollRef.current = true;
    viewport.scrollTo({
      top: viewport.scrollHeight,
      behavior,
    });
    window.setTimeout(() => {
      programmaticScrollRef.current = false;
      lastScrollTopRef.current = viewport.scrollTop;
    }, 0);
  }, []);

  const maybeResumeSticky = useCallback((viewport: HTMLDivElement) => {
    const distanceToBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
    if (distanceToBottom <= Math.max(RESUME_STICKY_PX, viewport.clientHeight)) {
      shouldStickToBottomRef.current = true;
      setShowScrollToBottom(false);
      scrollViewportToBottom(viewport);
    }
  }, [scrollViewportToBottom]);

  useEffect(() => {
    if (!shouldStickToBottomRef.current) {
      return;
    }
    const viewport = messagesViewportRef.current;
    if (!viewport) {
      return;
    }
    const frame = requestAnimationFrame(() => {
      if (!shouldStickToBottomRef.current) {
        return;
      }
      scrollViewportToBottom(viewport, streaming ? "auto" : "smooth");
    });
    return () => cancelAnimationFrame(frame);
  }, [messages, scrollViewportToBottom, streaming]);

  const activeSession = sessions.find((session) => session.session_id === currentSessionId);
  const isAnsweringAsk = Boolean(currentPendingHitl) && Boolean(latestAsk(messages));
  const activeSessionTitle = activeSession ? sessionTitle(activeSession) : titleFromMessages(messages);

  async function handleNewSession() {
    const created = await createSession();
    shouldStickToBottomRef.current = true;
    setShowScrollToBottom(false);
    setSessionMessages(created.session_id, () => created.messages ?? []);
    setCurrentSessionId(created.session_id);
    await refreshSessions();
  }

  function handleOpenSkills() {
    setSkillsOpen(true);
    void refreshSkills();
  }

  async function handleDeleteSession(sessionId: string) {
    abortControllers.current.get(sessionId)?.abort();
    abortControllers.current.delete(sessionId);
    markStreaming(sessionId, false);
    setSessionPendingHitl(sessionId, null);
    setMessagesBySession((prev) => {
      if (!(sessionId in prev)) {
        return prev;
      }
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
    await deleteSession(sessionId);
    const nextSessions = await refreshSessions();
    if (sessionId === currentSessionId) {
      if (nextSessions.length) {
        await loadSession(nextSessions[0].session_id);
      } else {
        await handleNewSession();
      }
    }
  }

  async function handleStop() {
    const targetSessionId = currentSessionId;
    if (!targetSessionId) {
      return;
    }
    try {
      await cancelSession(targetSessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      abortControllers.current.get(targetSessionId)?.abort();
    }
  }

  async function submitText(rawText: string) {
    const text = rawText.trim();
    if (!text || streaming) {
      return;
    }

    setInput("");
    setError(null);
    const controller = new AbortController();

    let activeSessionId = currentSessionId;
    let pendingBefore: PendingHitl | null = null;

    try {
      if (!activeSessionId) {
        const created = await createSession();
        activeSessionId = created.session_id;
        setSessionMessages(activeSessionId, () => created.messages ?? []);
        setCurrentSessionId(activeSessionId);
      }

      const sessionId = activeSessionId;
      abortControllers.current.set(sessionId, controller);
      markStreaming(sessionId, true);
      pendingBefore = pendingHitlBySession[sessionId] ?? null;

      shouldStickToBottomRef.current = true;
      setShowScrollToBottom(false);
      setSessionMessages(sessionId, (prev) => [
        ...prev,
        { role: "user", content: text },
        { role: "assistant", content: "", events: [] },
      ]);
      setSessions((prev) => {
        const optimisticTitle = text.slice(0, 60);
        const now = new Date().toISOString();
        let matched = false;
        const next = prev.map((session) => {
          if (session.session_id !== sessionId) {
            return session;
          }
          matched = true;
          return {
            ...session,
            title: shouldReplaceSessionTitle(session.title) ? optimisticTitle : session.title,
            status: "running",
            running: true,
            updated_at: now,
          };
        });
        if (matched) {
          return next;
        }
        return [
          {
            session_id: sessionId,
            title: optimisticTitle,
            created_at: now,
            updated_at: now,
            status: "running",
            running: true,
            message_count: 2,
          },
          ...next,
        ];
      });

      const streamInput = pendingBefore
        ? [
            {
              type: "function_call_output",
              call_id: pendingBefore.call_id,
              output: text,
            } as Record<string, unknown>,
          ]
        : text;
      const previousResponseId = pendingBefore?.response_id;

      // Optimistically clear the pending HITL — onTerminal/onRequiresAction
      // will reinstate or finalize it as the SSE drains.
      if (pendingBefore) {
        setSessionPendingHitl(sessionId, null);
      }

      await streamResponses(
        {
          sessionId,
          input: streamInput,
          previousResponseId,
          signal: controller.signal,
        },
        {
          onUpdate: ({ sessionId: eventSessionId, update }) => {
            setSessionMessages(eventSessionId, (prev) => applyAssistantUpdate(prev, update));
          },
          onRequiresAction: (pending) => {
            setSessionPendingHitl(sessionId, pending);
          },
        },
      );

      const detail = await getSession(sessionId);
      setSessionMessages(detail.session_id, (prev) => mergePostStreamMessages(prev, detail));
      setSessionPendingHitl(detail.session_id, detail.pending_hitl ?? null);
      await refreshSessions();
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        if (activeSessionId) {
          if (pendingBefore) {
            setSessionPendingHitl(activeSessionId, pendingBefore);
          }
          setSessionMessages(activeSessionId, (prev) =>
            applyAssistantUpdate(prev, {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: `**Error:** ${message}` },
            }),
          );
        }
      }
    } finally {
      if (activeSessionId) {
        markStreaming(activeSessionId, false);
        if (abortControllers.current.get(activeSessionId) === controller) {
          abortControllers.current.delete(activeSessionId);
        }
      }
    }
  }

  async function handleRegenerateLastAnswer() {
    if (!currentSessionId || streaming || isAnsweringAsk || !canRegenerateLastAssistant(messages)) {
      return;
    }

    setError(null);
    const controller = new AbortController();
    const activeSessionId = currentSessionId;
    abortControllers.current.set(activeSessionId, controller);
    markStreaming(activeSessionId, true);

    try {
      shouldStickToBottomRef.current = true;
      setShowScrollToBottom(false);
      setSessionMessages(activeSessionId, (prev) => {
        if (!canRegenerateLastAssistant(prev)) {
          return prev;
        }
        return [...prev.slice(0, -1), { role: "assistant", content: "", events: [] }];
      });
      setSessions((prev) =>
        prev.map((session) =>
          session.session_id === activeSessionId
            ? { ...session, status: "running", running: true, updated_at: new Date().toISOString() }
            : session,
        ),
      );

      await streamRegenerate(
        activeSessionId,
        {
          onUpdate: ({ sessionId: eventSessionId, update }) => {
            setSessionMessages(eventSessionId, (prev) => applyAssistantUpdate(prev, update));
          },
          onRequiresAction: (pending) => {
            setSessionPendingHitl(activeSessionId, pending);
          },
        },
        controller.signal,
      );

      const detail = await getSession(activeSessionId);
      setSessionMessages(detail.session_id, (prev) => mergePostStreamMessages(prev, detail));
      setSessionPendingHitl(detail.session_id, detail.pending_hitl ?? null);
      await refreshSessions();
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        try {
          const detail = await getSession(activeSessionId);
          setSessionMessages(detail.session_id, (prev) => mergePostStreamMessages(prev, detail));
          setSessionPendingHitl(detail.session_id, detail.pending_hitl ?? null);
        } catch {
          setSessionMessages(activeSessionId, (prev) =>
            applyAssistantUpdate(prev, {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: `**Error:** ${message}` },
            }),
          );
        }
      }
    } finally {
      markStreaming(activeSessionId, false);
      if (abortControllers.current.get(activeSessionId) === controller) {
        abortControllers.current.delete(activeSessionId);
      }
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    await submitText(input);
  }

  return (
    <div className="grid h-[100dvh] overflow-hidden bg-transparent lg:grid-cols-[320px_1fr]">
      <aside className="hidden h-[100dvh] border-r border-slate-200/80 bg-white/80 backdrop-blur lg:flex lg:flex-col">
        <div className="flex items-center gap-3 px-6 py-5">
          <div className="grid h-11 w-11 place-items-center rounded-xl bg-blue-600 text-white shadow-sm">
            <Bot className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-base font-semibold text-slate-900">PAI Assistant</h1>
            <p className="text-sm text-slate-500">Next.js client</p>
          </div>
        </div>
        <div className="space-y-2 px-4 pb-4">
          <Button className="w-full justify-start gap-2 rounded-lg" onClick={() => void handleNewSession()}>
            <Plus className="h-4 w-4" />
            New Task
          </Button>
          <Button
            className="w-full justify-start gap-2 rounded-lg border-blue-200 bg-blue-50/70 text-blue-700 hover:bg-blue-100 hover:text-blue-800"
            variant="outline"
            onClick={handleOpenSkills}
          >
            <BookOpen className="h-4 w-4" />
            <span className="flex-1 text-left">Skills</span>
            <span className="flex items-center gap-1" aria-hidden="true">
              <span className="h-2 w-2 rounded-full bg-blue-500" />
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
            </span>
          </Button>
        </div>
        <ScrollArea className="flex-1 px-3 [&>[data-radix-scroll-area-viewport]>div]:!block">
          <div className="w-full min-w-0 space-y-2 pb-4">
            {sessions.map((session) => (
              <div className="flex w-full min-w-0 items-center gap-2" key={session.session_id}>
                <button
                  className={cn(
                    "flex min-w-0 flex-1 items-center justify-between rounded-lg px-3 py-3 text-left text-sm transition-colors",
                    session.session_id === currentSessionId
                      ? "bg-blue-50 text-blue-700"
                      : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
                  )}
                  onClick={() => void loadSession(session.session_id)}
                >
                  <span className="truncate">{sessionTitle(session)}</span>
                  {session.running ? <Badge variant="outline">running</Badge> : null}
                </button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="shrink-0 text-slate-500 hover:bg-red-50 hover:text-red-600"
                  aria-label="Delete session"
                  title="Delete session"
                  onClick={() => void handleDeleteSession(session.session_id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        </ScrollArea>
        <div className="px-4 pb-5 pt-4">
          <Separator className="mb-4" />
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span className="inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500 shadow-[0_0_0_6px_rgba(34,197,94,0.12)]" />
            <span>Local session store</span>
          </div>
        </div>
      </aside>

      <main className="relative flex h-[100dvh] min-h-0 min-w-0 flex-col overflow-hidden">
        <header className="shrink-0 border-b border-slate-200/80 bg-white/75 px-5 py-4 backdrop-blur lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="max-w-[60vw] truncate text-lg font-semibold text-slate-900">
                {currentSessionId ? activeSessionTitle : "Connecting"}
              </h2>
              <p className="text-sm text-slate-500">{loading ? "Loading sessions..." : `${messages.length} messages`}</p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setModelDialogOpen(true)}
                className="rounded-md transition-opacity hover:opacity-80 focus:outline-none focus:ring-2 focus:ring-blue-300"
                aria-label="切换模型"
                title="点击切换模型"
              >
                <Badge variant="outline" className="cursor-pointer">
                  Model: {activeModel || "…"}
                </Badge>
              </button>
              <Badge variant="outline">Local</Badge>
              {streaming ? <Badge>Streaming</Badge> : <Badge variant="secondary">Idle</Badge>}
              <Button variant="outline" disabled={!streaming} onClick={() => void handleStop()}>
                <Square className="h-4 w-4" />
                Stop
              </Button>
            </div>
          </div>
        </header>

        {error ? (
          <div className="shrink-0 px-5 pt-4 lg:px-8">
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
          </div>
        ) : null}

        <div
          className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden"
          ref={messagesViewportRef}
          onWheel={(event) => {
            if (event.deltaY < 0) {
              shouldStickToBottomRef.current = false;
              setShowScrollToBottom(true);
            } else if (event.deltaY > 0) {
              maybeResumeSticky(event.currentTarget);
            }
          }}
          onTouchStart={(event) => {
            lastTouchYRef.current = event.touches[0]?.clientY ?? null;
          }}
          onTouchMove={(event) => {
            const currentY = event.touches[0]?.clientY ?? null;
            const previousY = lastTouchYRef.current;
            if (currentY !== null && previousY !== null && currentY > previousY + 2) {
              shouldStickToBottomRef.current = false;
              setShowScrollToBottom(true);
            } else if (currentY !== null && previousY !== null && currentY < previousY - 2) {
              maybeResumeSticky(event.currentTarget);
            }
            lastTouchYRef.current = currentY;
          }}
          onScroll={(event) => {
            const target = event.currentTarget;
            if (programmaticScrollRef.current) {
              lastScrollTopRef.current = target.scrollTop;
              return;
            }
            const distanceToBottom = target.scrollHeight - target.scrollTop - target.clientHeight;
            const scrollingUp = target.scrollTop < lastScrollTopRef.current - 2;
            const scrollingDown = target.scrollTop > lastScrollTopRef.current + 2;
            if (scrollingUp) {
              shouldStickToBottomRef.current = false;
              setShowScrollToBottom(true);
            } else if (distanceToBottom < STICKY_BOTTOM_PX) {
              shouldStickToBottomRef.current = true;
              setShowScrollToBottom(false);
            } else if (scrollingDown) {
              maybeResumeSticky(target);
            }
            lastScrollTopRef.current = target.scrollTop;
          }}
        >
          <section className="flex w-full min-w-0 flex-col gap-4 px-4 py-4 lg:px-8">
            {!messages.length && !loading ? (
              <div className="mx-auto flex min-h-[50vh] max-w-2xl flex-col items-center justify-center text-center">
                <div className="mb-6 grid h-16 w-16 place-items-center rounded-2xl bg-blue-600 text-white shadow-panel">
                  <Bot className="h-8 w-8" />
                </div>
                <h3 className="text-3xl font-semibold text-slate-900">How can I help?</h3>
                <p className="mt-3 max-w-xl text-sm text-slate-500">
                  Ask a question, run a task, or continue an agent workflow with the current session state.
                </p>
              </div>
            ) : null}

            {messages.map((message, index) => (
              <article
                className={cn(
                  "flex w-full min-w-0 items-start gap-3",
                  message.role === "user" ? "justify-end" : "justify-start",
                )}
                key={`${message.role}-${index}`}
              >
                {message.role !== "user" ? (
                  <div className="mt-1 hidden h-9 w-9 shrink-0 place-items-center rounded-xl bg-slate-900 text-white shadow-sm sm:grid">
                    <Bot className="h-4 w-4" />
                  </div>
                ) : null}

                <Card
                  className={cn(
                    "min-w-0 border shadow-sm",
                    message.role === "user"
                      ? "max-w-[min(78%,46rem)] border-blue-200 bg-blue-600 text-white"
                      : "flex-1 border-slate-200 bg-white/90 shadow-panel",
                  )}
                >
                  <CardContent className={message.role === "user" ? "px-3 py-2 sm:px-3 sm:py-2" : "p-3 sm:p-4"}>
                    {message.role === "user" ? (
                      <div className="flex min-w-0 items-center gap-2.5">
                        <div className="grid h-6 w-6 shrink-0 place-items-center rounded-md bg-white/15 text-white sm:hidden">
                          <User2 className="h-3.5 w-3.5" />
                        </div>
                        <div className="prose-agent min-w-0 max-w-none overflow-hidden break-words text-sm leading-relaxed text-white [&_*]:my-0 [&_*]:text-inherit">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{escapeModelProtocolTags(message.content)}</ReactMarkdown>
                        </div>
                      </div>
                    ) : (
                      <>
                        <MessageContent
                          message={message}
                          onSelectCandidate={(_, candidateIndex) => void submitText(String(candidateIndex + 1))}
                          streaming={streaming && index === messages.length - 1 && message.role === "assistant"}
                        />
                        {index === messages.length - 1 && canRegenerateLastAssistant(messages) && !isAnsweringAsk ? (
                          <div className="mt-3 flex justify-end">
                            <Button
                              className="h-8 gap-1.5 px-2.5 text-xs text-slate-600"
                              disabled={streaming || !currentSessionId}
                              onClick={() => void handleRegenerateLastAnswer()}
                              title="Regenerate answer"
                              type="button"
                              variant="ghost"
                            >
                              <RefreshCw className="h-3.5 w-3.5" />
                              Regenerate
                            </Button>
                          </div>
                        ) : null}
                      </>
                    )}
                  </CardContent>
                </Card>

                {message.role === "user" ? (
                  <div className="mt-1 hidden h-9 w-9 shrink-0 place-items-center rounded-xl bg-blue-600 text-white shadow-sm sm:grid">
                    <User2 className="h-4 w-4" />
                  </div>
                ) : null}
              </article>
            ))}

            {loading ? (
              <div className="flex items-center gap-3 text-sm text-slate-500">
                <LoaderCircle className="h-4 w-4 animate-spin" />
                Loading sessions
              </div>
            ) : null}
          </section>
        </div>

        {showScrollToBottom ? (
          <div className="pointer-events-none absolute inset-x-0 bottom-24 z-10 flex justify-center">
            <Button
              className="pointer-events-auto h-9 gap-1.5 rounded-full border-slate-200 bg-white px-3 text-xs text-slate-700 shadow-panel hover:bg-slate-50"
              type="button"
              variant="outline"
              onClick={() => {
                const viewport = messagesViewportRef.current;
                shouldStickToBottomRef.current = true;
                setShowScrollToBottom(false);
                if (viewport) {
                  scrollViewportToBottom(viewport);
                }
              }}
            >
              <ChevronDown className="h-3.5 w-3.5" />
              Latest
            </Button>
          </div>
        ) : null}

        <div className="shrink-0 border-t border-slate-200/80 bg-white/80 px-4 py-4 backdrop-blur lg:px-8">
          <form className="flex w-full min-w-0 items-end gap-3" onSubmit={(event) => void handleSubmit(event)}>
            <Textarea
              className="max-h-40 min-h-[56px] resize-none rounded-xl bg-white shadow-sm"
              value={input}
              disabled={streaming}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void handleSubmit(event);
                }
              }}
              placeholder={isAnsweringAsk ? "Answer the question..." : "Enter a task..."}
              rows={1}
            />
            <Button className="h-14 rounded-xl px-5" disabled={!input.trim() || streaming} type="submit">
              Send
            </Button>
          </form>
        </div>
      </main>
      {skillsOpen ? (
        <SkillsDialog
          inventory={skillsInventory}
          loading={skillsLoading}
          error={skillsError}
          onClose={() => setSkillsOpen(false)}
        />
      ) : null}
      {modelDialogOpen ? (
        <ModelPickerDialog
          current={activeModel}
          onClose={() => setModelDialogOpen(false)}
          onSaved={(name) => {
            setActiveModelState(name);
            try {
              localStorage.setItem(ACTIVE_MODEL_LS_KEY, name);
            } catch {
              // ignore
            }
          }}
        />
      ) : null}
    </div>
  );
}
