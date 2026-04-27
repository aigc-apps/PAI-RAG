"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Bot, Brain, CheckCircle2, ChevronDown, LoaderCircle, LogOut, Plus, Square, Trash2, User2, Wrench } from "lucide-react";
import {
  cancelSession,
  clearAuthToken,
  createSession,
  currentUser,
  deleteSession,
  getAuthToken,
  getSession,
  listSessions,
  login as loginUser,
  register as registerUser,
  setAuthToken,
  streamAgentPrompt,
} from "@/lib/api";
import type { AgentUpdate, AskUserPayload, ChatMessage, SessionSummary, UserProfile } from "@/lib/types";
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

const INTERNAL_TOOL_NAMES = new Set(["update_working_checkpoint", "start_long_term_update"]);

function Markdown({ content }: { content: string }) {
  return (
    <div className="prose-agent">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
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
        {item.kind === "execute" ? <CodeRunDetails item={item} /> : <GenericToolDetails item={item} />}
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
        <AgentReasoningContent item={group.thought} />
        {group.items.map((item, index) => (
          <ProcessSubStep item={item} key={item.id} stepNumber={index + 1} />
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
  const totalItems = groups.reduce((total, group) => total + group.items.length, 0);

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
  const hasContent = Boolean(displayContent);

  if (!hasContent && !processItems.length && !asks.length) {
    return streaming ? <span className="inline-flex h-4 w-1 animate-pulse rounded bg-blue-500" /> : null;
  }

  return (
    <div className="space-y-3">
      {processItems.length ? <ProcessBlock items={processItems} streaming={streaming} /> : null}
      {hasContent ? <Markdown content={displayContent} /> : null}
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

function AuthPanel({
  mode,
  username,
  password,
  error,
  loading,
  onModeChange,
  onUsernameChange,
  onPasswordChange,
  onSubmit,
}: {
  mode: "login" | "register";
  username: string;
  password: string;
  error: string | null;
  loading: boolean;
  onModeChange: (mode: "login" | "register") => void;
  onUsernameChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onSubmit: () => void;
}) {
  return (
    <div className="grid min-h-[100dvh] place-items-center bg-slate-50 px-4">
      <Card className="w-full max-w-md border-slate-200 bg-white shadow-panel">
        <CardContent className="space-y-5 p-6">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-xl bg-blue-600 text-white shadow-sm">
              <Bot className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-slate-900">PAI Assistant</h1>
              <p className="text-sm text-slate-500">{mode === "login" ? "Sign in to continue" : "Create an account"}</p>
            </div>
          </div>

          <div className="grid grid-cols-2 rounded-lg bg-slate-100 p-1">
            <button
              className={cn(
                "rounded-md px-3 py-2 text-sm font-medium transition-colors",
                mode === "login" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-900",
              )}
              onClick={() => onModeChange("login")}
              type="button"
            >
              Login
            </button>
            <button
              className={cn(
                "rounded-md px-3 py-2 text-sm font-medium transition-colors",
                mode === "register" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-900",
              )}
              onClick={() => onModeChange("register")}
              type="button"
            >
              Register
            </button>
          </div>

          <form
            className="space-y-3"
            onSubmit={(event) => {
              event.preventDefault();
              onSubmit();
            }}
          >
            <label className="block space-y-1.5">
              <span className="text-sm font-medium text-slate-700">Username</span>
              <input
                autoComplete="username"
                className="h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm outline-none transition-colors focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                onChange={(event) => onUsernameChange(event.target.value)}
                placeholder="username"
                value={username}
              />
            </label>
            <label className="block space-y-1.5">
              <span className="text-sm font-medium text-slate-700">Password</span>
              <input
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                className="h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm outline-none transition-colors focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                onChange={(event) => onPasswordChange(event.target.value)}
                placeholder="password"
                type="password"
                value={password}
              />
            </label>
            {error ? <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div> : null}
            <Button className="h-11 w-full rounded-lg" disabled={loading || !username.trim() || !password} type="submit">
              {loading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
              {mode === "login" ? "Login" : "Register"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

export function ChatShell() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const refreshSessions = useCallback(async () => {
    const data = await listSessions();
    setSessions(data);
    return data;
  }, []);

  const loadSession = useCallback(
    async (sessionId: string) => {
      const detail = await getSession(sessionId);
      setCurrentSessionId(detail.session_id);
      setMessages(detail.messages ?? []);
      await refreshSessions();
    },
    [refreshSessions],
  );

  const loadInitialSession = useCallback(async () => {
    const data = await refreshSessions();
    if (data.length) {
      await loadSession(data[0].session_id);
    } else {
      const created = await createSession();
      setCurrentSessionId(created.session_id);
      setMessages(created.messages ?? []);
      await refreshSessions();
    }
  }, [loadSession, refreshSessions]);

  useEffect(() => {
    let active = true;
    async function boot() {
      try {
        setError(null);
        const token = getAuthToken();
        if (!token) {
          setAuthReady(true);
          setLoading(false);
          return;
        }
        const profile = await currentUser();
        if (!active) {
          return;
        }
        setUser(profile);
        await loadInitialSession();
      } catch (err) {
        clearAuthToken();
        setUser(null);
        setAuthError("Session expired. Please login again.");
      } finally {
        if (active) {
          setAuthReady(true);
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
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, streaming]);

  const isAnsweringAsk = Boolean(latestAsk(messages));

  async function handleAuthSubmit() {
    if (authLoading) {
      return;
    }
    setAuthLoading(true);
    setAuthError(null);
    setError(null);
    try {
      const payload =
        authMode === "login"
          ? await loginUser(authUsername, authPassword)
          : await registerUser(authUsername, authPassword);
      setAuthToken(payload.access_token);
      setUser(payload.user);
      setAuthPassword("");
      setSessions([]);
      setCurrentSessionId(null);
      setMessages([]);
      setLoading(true);
      await loadInitialSession();
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
      setAuthLoading(false);
    }
  }

  function handleLogout() {
    abortRef.current?.abort();
    clearAuthToken();
    setUser(null);
    setSessions([]);
    setCurrentSessionId(null);
    setMessages([]);
    setInput("");
    setStreaming(false);
    setError(null);
    setAuthError(null);
  }

  async function handleNewSession() {
    const created = await createSession();
    setCurrentSessionId(created.session_id);
    setMessages(created.messages ?? []);
    await refreshSessions();
  }

  async function handleDeleteSession(sessionId: string) {
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
    if (currentSessionId) {
      await cancelSession(currentSessionId);
    }
    abortRef.current?.abort();
    setStreaming(false);
  }

  async function submitText(rawText: string) {
    const text = rawText.trim();
    if (!text || streaming) {
      return;
    }

    setInput("");
    setError(null);
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;

    let activeSessionId = currentSessionId;

    try {
      if (!activeSessionId) {
        const created = await createSession();
        activeSessionId = created.session_id;
        setCurrentSessionId(activeSessionId);
      }

      setMessages((prev) => [...prev, { role: "user", content: text }, { role: "assistant", content: "", events: [] }]);

      const returnedSessionId = await streamAgentPrompt(
        activeSessionId,
        text,
        ({ sessionId, update }) => {
          activeSessionId = sessionId;
          setCurrentSessionId(sessionId);
          setMessages((prev) => applyAssistantUpdate(prev, update));
        },
        controller.signal,
      );

      activeSessionId = returnedSessionId || activeSessionId;
      if (activeSessionId) {
        setCurrentSessionId(activeSessionId);
        const detail = await getSession(activeSessionId);
        setMessages(detail.messages ?? []);
      }
      await refreshSessions();
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        setMessages((prev) =>
          applyAssistantUpdate(prev, {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: `**Error:** ${message}` },
          }),
        );
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    await submitText(input);
  }

  if (!authReady) {
    return (
      <div className="grid min-h-[100dvh] place-items-center bg-slate-50 text-sm text-slate-500">
        <span className="inline-flex items-center gap-2">
          <LoaderCircle className="h-4 w-4 animate-spin" />
          Loading account
        </span>
      </div>
    );
  }

  if (!user) {
    return (
      <AuthPanel
        mode={authMode}
        username={authUsername}
        password={authPassword}
        error={authError}
        loading={authLoading}
        onModeChange={(nextMode) => {
          setAuthMode(nextMode);
          setAuthError(null);
        }}
        onUsernameChange={setAuthUsername}
        onPasswordChange={setAuthPassword}
        onSubmit={() => void handleAuthSubmit()}
      />
    );
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
        <div className="px-4 pb-4">
          <Button className="w-full justify-start gap-2 rounded-lg" onClick={() => void handleNewSession()}>
            <Plus className="h-4 w-4" />
            New Task
          </Button>
        </div>
        <ScrollArea className="flex-1 px-3">
          <div className="space-y-2 pb-4">
            {sessions.map((session) => (
              <div className="flex items-center gap-2" key={session.session_id}>
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
                  className="text-slate-400 hover:bg-red-50 hover:text-red-600"
                  disabled={session.session_id === currentSessionId && sessions.length <= 1}
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
          <div className="mb-3 flex items-center justify-between gap-3 rounded-lg bg-slate-50 px-3 py-2">
            <div className="min-w-0">
              <div className="truncate text-sm font-medium text-slate-800">{user.username}</div>
              <div className="text-xs text-slate-500">Signed in</div>
            </div>
            <Button className="shrink-0 text-slate-500 hover:text-slate-900" onClick={handleLogout} size="icon" variant="ghost">
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span className="inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500 shadow-[0_0_0_6px_rgba(34,197,94,0.12)]" />
            <span>Next proxy -&gt; FastAPI backend</span>
          </div>
        </div>
      </aside>

      <main className="flex h-[100dvh] min-h-0 flex-col overflow-hidden">
        <header className="shrink-0 border-b border-slate-200/80 bg-white/75 px-5 py-4 backdrop-blur lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">{currentSessionId ? "Chat" : "Connecting"}</h2>
              <p className="text-sm text-slate-500">{loading ? "Loading sessions..." : `${messages.length} messages`}</p>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline">{user.username}</Badge>
              <Button className="lg:hidden" onClick={handleLogout} size="icon" variant="ghost">
                <LogOut className="h-4 w-4" />
              </Button>
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

        <ScrollArea className="min-h-0 flex-1">
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
                  "flex w-full min-w-0 gap-3",
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
                      ? "max-w-[78%] border-blue-200 bg-blue-600 text-white"
                      : "w-full border-slate-200 bg-white/90 shadow-panel",
                  )}
                >
                  <CardContent className={message.role === "user" ? "px-3 py-2 sm:px-3 sm:py-2" : "p-3 sm:p-4"}>
                    {message.role === "user" ? (
                      <div className="flex items-center gap-2.5">
                        <div className="grid h-6 w-6 shrink-0 place-items-center rounded-md bg-white/15 text-white sm:hidden">
                          <User2 className="h-3.5 w-3.5" />
                        </div>
                        <div className="prose-agent max-w-none text-sm leading-relaxed text-white [&_*]:my-0 [&_*]:text-inherit">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                        </div>
                      </div>
                    ) : (
                      <MessageContent
                        message={message}
                        onSelectCandidate={(_, candidateIndex) => void submitText(String(candidateIndex + 1))}
                        streaming={streaming && index === messages.length - 1 && message.role === "assistant"}
                      />
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
            <div ref={bottomRef} />
          </section>
        </ScrollArea>

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
    </div>
  );
}
