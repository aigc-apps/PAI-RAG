import type {
  AgentUpdate,
  PendingHitl,
  ResponsesSseEvent,
  SessionDetail,
  SessionSummary,
  SkillInventory,
  StreamEvent,
} from "@/lib/types";

function assertOk(response: Response) {
  if (response.ok) {
    return Promise.resolve();
  }
  return response.text().then((text) => {
    throw new Error(`${response.status}: ${text.slice(0, 600)}`);
  });
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await fetch("/api/sessions", { cache: "no-store" });
  await assertOk(response);
  const payload = await response.json();
  return payload.data ?? [];
}

export async function createSession(): Promise<SessionDetail> {
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  await assertOk(response);
  return response.json();
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  const response = await fetch(`/api/sessions/${sessionId}`, { cache: "no-store" });
  await assertOk(response);
  return response.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
  await assertOk(response);
}

export async function cancelSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}/cancel`, { method: "POST" });
  await assertOk(response);
}

export type ModelInfo = { id: string; active: boolean };
export type ModelsResponse = { active_model: string; data: ModelInfo[] };

export async function getActiveModel(): Promise<ModelsResponse> {
  const response = await fetch("/api/models", { cache: "no-store" });
  await assertOk(response);
  return response.json();
}

export async function setActiveModel(name: string): Promise<{ active_model: string }> {
  const response = await fetch("/api/models/active", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: name }),
  });
  await assertOk(response);
  return response.json();
}

export async function getSkills(): Promise<SkillInventory> {
  const response = await fetch("/api/skills", { cache: "no-store" });
  await assertOk(response);
  const payload = await response.json();
  return {
    official: payload.official ?? [],
    evolved: payload.evolved ?? [],
  };
}

// ─── /v1/responses streaming ──────────────────────────────────────────

export type ResponsesInputItem = Record<string, unknown>;

export interface StreamResponsesOptions {
  sessionId: string;
  input: string | ResponsesInputItem[];
  previousResponseId?: string;
  signal?: AbortSignal;
  model?: string;
  store?: boolean;
  allowHitl?: boolean;
}

export interface ResponsesStreamHandlers {
  onUpdate: (event: StreamEvent) => void;
  onRequiresAction?: (pending: PendingHitl) => void;
  onResponseId?: (responseId: string) => void;
  onTerminal?: (kind: "completed" | "failed", payload: Record<string, unknown>) => void;
}

export async function streamResponses(
  opts: StreamResponsesOptions,
  handlers: ResponsesStreamHandlers,
): Promise<string | undefined> {
  const body: Record<string, unknown> = {
    conversation: opts.sessionId,
    input: opts.input,
    stream: true,
  };
  if (opts.previousResponseId) {
    body.previous_response_id = opts.previousResponseId;
  }
  if (opts.model) {
    body.model = opts.model;
  }
  if (opts.store !== undefined) {
    body.store = opts.store;
  }
  if (opts.allowHitl !== undefined) {
    body.allow_hitl = opts.allowHitl;
  }
  const response = await fetch("/api/responses", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  await assertOk(response);
  await consumeResponsesSse(response, opts.sessionId, handlers);
  return opts.sessionId;
}

export async function streamRegenerate(
  sessionId: string,
  handlers: ResponsesStreamHandlers,
  signal?: AbortSignal,
): Promise<string | undefined> {
  const response = await fetch(`/api/sessions/${sessionId}/regenerate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({}),
    signal,
  });
  await assertOk(response);
  await consumeResponsesSse(response, sessionId, handlers);
  return sessionId;
}

async function consumeResponsesSse(
  response: Response,
  sessionId: string,
  handlers: ResponsesStreamHandlers,
): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) {
    return;
  }
  const decoder = new TextDecoder();
  let buffer = "";
  const state: ResponsesParseState = createParseState();

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseSseFrame(chunk);
      if (event) {
        dispatchResponsesEvent(event, state, sessionId, handlers);
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
  if (buffer.trim()) {
    const event = parseSseFrame(buffer);
    if (event) {
      dispatchResponsesEvent(event, state, sessionId, handlers);
    }
  }
}

function parseSseFrame(frame: string): ResponsesSseEvent | null {
  const lines = frame.split("\n");
  let eventName = "";
  const dataLines: string[] = [];
  for (const rawLine of lines) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (!dataLines.length) {
    return null;
  }
  const data = dataLines.join("\n");
  if (data === "[DONE]") {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") {
    return null;
  }
  const record = parsed as Record<string, unknown>;
  const type = eventName || (typeof record.type === "string" ? record.type : "");
  if (!type) {
    return null;
  }
  return { type, data: record };
}

function dispatchResponsesEvent(
  event: ResponsesSseEvent,
  state: ResponsesParseState,
  sessionId: string,
  handlers: ResponsesStreamHandlers,
) {
  if (event.type === "response.created") {
    const id = stringField(event.data, "id");
    if (id) {
      state.responseId = id;
      handlers.onResponseId?.(id);
    }
    return;
  }
  if (event.type === "response.requires_action") {
    flushPendingThink(state, sessionId, handlers);
    const pending = parseRequiresAction(event.data);
    if (pending) {
      handlers.onRequiresAction?.(pending);
      dispatchUpdates(updatesForRequiresAction(pending), state, sessionId, handlers);
    }
    handlers.onTerminal?.("completed", event.data);
    return;
  }
  if (event.type === "response.completed") {
    const terminalText = responseOutputText(event.data);
    if (terminalText && !state.visibleTextEmitted) {
      dispatchUpdates(consumeTextDelta(state, terminalText), state, sessionId, handlers);
    }
    flushPendingThink(state, sessionId, handlers);
    handlers.onTerminal?.("completed", event.data);
    return;
  }
  if (event.type === "response.failed") {
    flushPendingThink(state, sessionId, handlers);
    const message = errorMessage(event.data) || "Run failed";
    dispatchUpdates(
      [{ sessionUpdate: "agent_message_chunk", content: { type: "text", text: `**Error:** ${message}` } }],
      state,
      sessionId,
      handlers,
    );
    handlers.onTerminal?.("failed", event.data);
    return;
  }
  const updates = responsesEventToUpdates(event, state);
  dispatchUpdates(updates, state, sessionId, handlers);
}

function dispatchUpdates(
  updates: AgentUpdate[],
  state: ResponsesParseState,
  sessionId: string,
  handlers: ResponsesStreamHandlers,
) {
  for (const update of updates) {
    if (update.sessionUpdate === "agent_message_chunk" && update.content.text) {
      state.visibleTextEmitted = true;
    }
    handlers.onUpdate({ sessionId, update });
  }
}

// ─── parser ───────────────────────────────────────────────────────────

interface ResponsesParseState {
  responseId: string;
  toolCallByItemId: Map<string, ToolCallEntry>;
  toolCallByCallId: Map<string, ToolCallEntry>;
  reasoningStarted: Set<string>;
  activeReasoningStepId: string;
  activeReasoningSynthetic: boolean;
  syntheticReasoningText: string;
  syntheticReasoningTextDone: boolean;
  toolCounter: number;
  // Inline private reasoning block tracking. The model emits these
  // tags inside `output_text.delta` chunks; we re-route the inner text to
  // synthetic `thought_*` events so the UI groups them in an Agent step
  // alongside tool calls (instead of leaking into the final answer).
  thinkBuffer: string;
  thinkId: string;
  thinkOpen: boolean;
  thinkCloseTag: string;
  thinkMode: PrivateTextMode;
  thinkCounter: number;
  visibleTextEmitted: boolean;
}

type PrivateTextMode = "thought" | "discard";

interface ToolCallEntry {
  toolCallId: string;
  callId: string;
  name: string;
  index: number;
  argumentsText: string;
  status: "pending" | "in_progress" | "completed" | "failed";
}

function createParseState(): ResponsesParseState {
  return {
    responseId: "",
    toolCallByItemId: new Map(),
    toolCallByCallId: new Map(),
    reasoningStarted: new Set(),
    activeReasoningStepId: "",
    activeReasoningSynthetic: false,
    syntheticReasoningText: "",
    syntheticReasoningTextDone: false,
    toolCounter: 0,
    thinkBuffer: "",
    thinkId: "",
    thinkOpen: false,
    thinkCloseTag: "",
    thinkMode: "thought",
    thinkCounter: 0,
    visibleTextEmitted: false,
  };
}

const THINK_TAGS = [
  ["<clinical-thinking>", "</clinical-thinking>"],
  ["<clinical_thinking>", "</clinical_thinking>"],
  ["<taking-action>", "</taking-action>"],
  ["<taking_action>", "</taking_action>"],
  ["<skill-context>", "</skill-context>"],
  ["<skill_context>", "</skill_context>"],
  ["<thinking>", "</thinking>"],
  ["<checking>", "</checking>"],
  ["<taking>", "</taking>"],
  ["<working>", "</working>"],
] as const;

const PRIVATE_TEXT_TAGS: ReadonlyArray<{ openTag: string; closeTag: string; mode: PrivateTextMode }> = [
  { openTag: "<summary>", closeTag: "</summary>", mode: "discard" },
  { openTag: "<forcing_skill_activation>", closeTag: "</forcing_skill_activation>", mode: "discard" },
  ...THINK_TAGS.map(([openTag, closeTag]) => ({ openTag, closeTag, mode: "thought" as const })),
];

// Splits a text delta around model-private reasoning tags and emits
// either visible-content updates or thought_* updates. `thinkBuffer` carries
// any trailing partial tag (e.g. `<think`) across chunks. The first call
// after a close tag (or at start) routes plain text to `agent_message_chunk`;
// once a reasoning tag is seen, all bytes go to `thought_delta` until its close tag.
// `<summary>` is a protocol block and is discarded rather than shown.
function consumeTextDelta(state: ResponsesParseState, delta: string): AgentUpdate[] {
  const updates: AgentUpdate[] = [];
  let buffer = state.thinkBuffer + delta;
  state.thinkBuffer = "";

  while (buffer.length > 0) {
    if (state.thinkOpen) {
      const closeTag = state.thinkCloseTag || "</thinking>";
      const closeIdx = buffer.indexOf(closeTag);
      if (closeIdx === -1) {
        // Hold back any trailing partial close tag (e.g. "</thinkin") so we
        // don't emit it as visible content if it turns into a real tag next.
        const safeUpTo = clipPartialTagAtEnd(buffer, closeTag);
        if (safeUpTo > 0 && state.thinkMode === "thought") {
          updates.push({
            sessionUpdate: "thought_delta",
            thoughtId: state.thinkId,
            content: { type: "text", text: buffer.slice(0, safeUpTo) },
          });
        }
        state.thinkBuffer = buffer.slice(safeUpTo);
        return updates;
      }
      const inner = buffer.slice(0, closeIdx);
      if (inner && state.thinkMode === "thought") {
        updates.push({
          sessionUpdate: "thought_delta",
          thoughtId: state.thinkId,
          content: { type: "text", text: inner },
        });
      }
      if (state.thinkMode === "thought") {
        updates.push({
          sessionUpdate: "thought_done",
          thoughtId: state.thinkId,
          status: "completed",
        });
      }
      state.thinkOpen = false;
      state.thinkCloseTag = "";
      state.thinkMode = "thought";
      buffer = buffer.slice(closeIdx + closeTag.length);
      continue;
    }

    // Not currently inside an internal block. Look for the next open tag.
    const open = findNextThinkOpen(buffer);
    if (!open) {
      const safeUpTo = clipPartialOpenTagAtEnd(buffer);
      if (safeUpTo > 0) {
        updates.push({
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: buffer.slice(0, safeUpTo) },
        });
      }
      state.thinkBuffer = buffer.slice(safeUpTo);
      return updates;
    }
    const before = buffer.slice(0, open.index);
    if (before) {
      updates.push({
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: before },
      });
    }
    state.thinkCounter += 1;
    state.thinkId = `inline-think-${state.thinkCounter}`;
    state.thinkOpen = true;
    state.thinkCloseTag = open.closeTag;
    state.thinkMode = open.mode;
    if (open.mode === "thought") {
      updates.push({
        sessionUpdate: "thought_start",
        thoughtId: state.thinkId,
        title: "Thinking",
        status: "in_progress",
      });
    }
    buffer = buffer.slice(open.index + open.openTag.length);
  }
  return updates;
}

// If `buffer` ends with a strict prefix of `tag` (e.g. ends with "<think"
// when tag is "<thinking>"), return the index up to which we can safely
// emit; the remainder is held for the next chunk. Avoids splitting a
// real tag that happens to straddle a chunk boundary.
// On terminal events, drain any held buffer / open thinking block so
// trailing characters or an unclosed internal tag don't get swallowed.
function flushPendingThink(
  state: ResponsesParseState,
  sessionId: string,
  handlers: ResponsesStreamHandlers,
) {
  const { thinkBuffer, thinkOpen, thinkId, thinkMode } = state;
  state.thinkBuffer = "";
  if (thinkOpen) {
    if (thinkMode === "thought" && thinkBuffer) {
      handlers.onUpdate({
        sessionId,
        update: {
          sessionUpdate: "thought_delta",
          thoughtId: thinkId,
          content: { type: "text", text: thinkBuffer },
        },
      });
    }
    if (thinkMode === "thought") {
      handlers.onUpdate({
        sessionId,
        update: { sessionUpdate: "thought_done", thoughtId: thinkId, status: "completed" },
      });
    }
    state.thinkOpen = false;
    state.thinkId = "";
    state.thinkCloseTag = "";
    state.thinkMode = "thought";
    return;
  }
  if (thinkBuffer) {
    state.visibleTextEmitted = true;
    handlers.onUpdate({
      sessionId,
      update: { sessionUpdate: "agent_message_chunk", content: { type: "text", text: thinkBuffer } },
    });
  }
}

function findNextThinkOpen(buffer: string): { index: number; openTag: string; closeTag: string; mode: PrivateTextMode } | null {
  let best: { index: number; openTag: string; closeTag: string; mode: PrivateTextMode } | null = null;
  for (const { openTag, closeTag, mode } of PRIVATE_TEXT_TAGS) {
    const index = buffer.indexOf(openTag);
    if (index !== -1 && (!best || index < best.index)) {
      best = { index, openTag, closeTag, mode };
    }
  }
  return best;
}

function clipPartialOpenTagAtEnd(buffer: string): number {
  let safeUpTo = buffer.length;
  for (const { openTag } of PRIVATE_TEXT_TAGS) {
    safeUpTo = Math.min(safeUpTo, clipPartialTagAtEnd(buffer, openTag));
  }
  return safeUpTo;
}

function clipPartialTagAtEnd(buffer: string, tag: string): number {
  for (let n = Math.min(tag.length - 1, buffer.length); n > 0; n -= 1) {
    if (buffer.endsWith(tag.slice(0, n))) {
      return buffer.length - n;
    }
  }
  return buffer.length;
}

function toolKind(name: string) {
  if (name === "code_run") return "execute";
  if (name === "file_read") return "read";
  if (name === "file_write" || name === "file_patch") return "edit";
  if (name === "ask_user") return "ask";
  return "tool";
}

function consumeTextAsThought(
  state: ResponsesParseState,
  thoughtId: string,
  text: string,
): AgentUpdate[] {
  if (!text) return [];
  return consumeTextDelta(state, text).flatMap((update): AgentUpdate[] => {
    if (update.sessionUpdate === "agent_message_chunk" || update.sessionUpdate === "thought_delta") {
      const content = update.content.text;
      return content ? [{ sessionUpdate: "thought_delta", thoughtId, content: { type: "text", text: content } }] : [];
    }
    return [];
  });
}

function resetInlineTextParser(state: ResponsesParseState) {
  state.thinkBuffer = "";
  state.thinkId = "";
  state.thinkOpen = false;
  state.thinkCloseTag = "";
  state.thinkMode = "thought";
}

export function responsesEventToUpdates(
  event: ResponsesSseEvent,
  state: ResponsesParseState,
): AgentUpdate[] {
  const data = event.data;

  if (event.type === "response.reasoning_text.delta") {
    const delta = stringField(data, "delta");
    if (!delta) return [];
    const stepId = stringField(data, "step_id") || state.activeReasoningStepId;
    if (!stepId) return [];
    return consumeTextAsThought(state, stepId, delta);
  }

  if (event.type === "response.output_text.delta") {
    const delta = stringField(data, "delta");
    if (!delta) return [];
    if (state.activeReasoningSynthetic && state.activeReasoningStepId) {
      state.syntheticReasoningText += delta;
      return consumeTextAsThought(state, state.activeReasoningStepId, delta);
    }
    return consumeTextDelta(state, delta);
  }

  if (event.type === "response.output_text.done") {
    if (state.activeReasoningSynthetic && state.activeReasoningStepId) {
      const stepId = state.activeReasoningStepId;
      const text = stringField(data, "text") || state.syntheticReasoningText;
      state.syntheticReasoningText = "";
      state.syntheticReasoningTextDone = true;
      resetInlineTextParser(state);
      return [
        { sessionUpdate: "thought_done", thoughtId: stepId, status: "completed", hidden: true },
        ...consumeTextDelta(state, text),
      ];
    }
    return [];
  }

  if (event.type === "response.output_item.added") {
    const item = recordField(data, "item");
    if (!item) return [];
    const itemType = stringField(item, "type");
    if (itemType === "function_call") {
      const entry = registerToolCall(state, item);
      const hidden = isInternalToolName(entry.name);
      return [
        {
          sessionUpdate: "tool_call",
          toolCallId: entry.toolCallId,
          title: entry.name,
          name: entry.name,
          kind: toolKind(entry.name),
          status: "pending",
          hidden,
        },
      ];
    }
    if (itemType === "function_call_output") {
      const callId = stringField(item, "call_id");
      const entry = state.toolCallByCallId.get(callId);
      if (!entry) return [];
      entry.status = "completed";
      const output = stringField(item, "output");
      let parsed: unknown = output;
      try {
        parsed = JSON.parse(output);
      } catch {
        // keep raw string
      }
      return [
        {
          sessionUpdate: "tool_call_update",
          toolCallId: entry.toolCallId,
          status: "completed",
          content: typeof parsed === "string" ? { type: "text", text: parsed } : undefined,
          data: typeof parsed === "string" ? undefined : parsed,
        },
      ];
    }
    return [];
  }

  if (event.type === "response.output_item.done") {
    const item = recordField(data, "item");
    if (!item) return [];
    const itemType = stringField(item, "type");
    if (itemType === "function_call") {
      const itemId = stringField(item, "id");
      const entry = state.toolCallByItemId.get(itemId);
      if (!entry) return [];
      const argumentsText = stringField(item, "arguments") || entry.argumentsText;
      entry.argumentsText = argumentsText;
      entry.status = entry.status === "completed" ? entry.status : "in_progress";
      const parsed = parseJson(argumentsText);
      return [
        {
          sessionUpdate: "tool_call_delta",
          toolCallId: entry.toolCallId,
          index: entry.index,
          name: entry.name,
          kind: toolKind(entry.name),
          status: entry.status,
          hidden: isInternalToolName(entry.name),
          argumentsText,
        },
        ...(parsed
          ? [
              {
                sessionUpdate: "tool_call",
                toolCallId: entry.toolCallId,
                title: entry.name,
                name: entry.name,
                kind: toolKind(entry.name),
                status: entry.status,
                hidden: isInternalToolName(entry.name),
                input: parsed,
              } as AgentUpdate,
            ]
          : []),
      ];
    }
    return [];
  }

  if (event.type === "response.function_call_arguments.delta") {
    const itemId = stringField(data, "item_id");
    const entry = state.toolCallByItemId.get(itemId);
    if (!entry) return [];
    const delta = stringField(data, "delta");
    entry.argumentsText += delta;
    entry.status = "in_progress";
    return [
      {
        sessionUpdate: "tool_call_delta",
        toolCallId: entry.toolCallId,
        index: entry.index,
        name: entry.name,
        kind: toolKind(entry.name),
        status: "in_progress",
        hidden: isInternalToolName(entry.name),
        argumentsDelta: delta,
      },
    ];
  }

  if (event.type === "response.function_call_arguments.done") {
    const itemId = stringField(data, "item_id");
    const entry = state.toolCallByItemId.get(itemId);
    if (!entry) return [];
    const argumentsText = stringField(data, "arguments") || entry.argumentsText;
    entry.argumentsText = argumentsText;
    entry.status = "in_progress";
    return [
      {
        sessionUpdate: "tool_call_delta",
        toolCallId: entry.toolCallId,
        index: entry.index,
        name: entry.name,
        kind: toolKind(entry.name),
        status: "in_progress",
        hidden: isInternalToolName(entry.name),
        argumentsText,
      },
    ];
  }

  if (event.type === "response.reasoning_step.started") {
    const stepId = stringField(data, "step_id") || `rs_${state.toolCounter + 1}`;
    if (state.reasoningStarted.has(stepId)) {
      return [];
    }
    state.reasoningStarted.add(stepId);
    state.activeReasoningStepId = stepId;
    state.activeReasoningSynthetic = Boolean(data.synthetic);
    state.syntheticReasoningText = "";
    state.syntheticReasoningTextDone = false;
    return [
      {
        sessionUpdate: "thought_start",
        thoughtId: stepId,
        title: "Agent step",
        status: "in_progress",
      },
    ];
  }

  if (event.type === "response.reasoning_step.completed") {
    const stepId = stringField(data, "step_id");
    if (!stepId) return [];
    const updates: AgentUpdate[] = [];
    if (
      state.activeReasoningSynthetic &&
      state.activeReasoningStepId === stepId &&
      state.thinkBuffer &&
      !state.syntheticReasoningTextDone
    ) {
      updates.push({
        sessionUpdate: "thought_delta",
        thoughtId: stepId,
        content: { type: "text", text: state.thinkBuffer },
      });
      resetInlineTextParser(state);
    }
    if (state.activeReasoningStepId === stepId) {
      state.activeReasoningStepId = "";
      state.activeReasoningSynthetic = false;
      state.syntheticReasoningText = "";
      state.syntheticReasoningTextDone = false;
    }
    updates.push({
      sessionUpdate: "thought_done",
      thoughtId: stepId,
      status: "completed",
      hidden: Boolean(data.hidden),
    });
    return updates;
  }

  return [];
}

function updatesForRequiresAction(pending: PendingHitl): AgentUpdate[] {
  if (pending.tool_name === "ask_user") {
    return [
      {
        sessionUpdate: "ask_user",
        question: pending.question || "Please provide input.",
        candidates: pending.candidates || [],
      },
    ];
  }
  return [];
}

function parseRequiresAction(data: Record<string, unknown>): PendingHitl | null {
  const responseId = stringField(data, "id");
  const requiredAction = recordField(data, "required_action");
  if (!requiredAction) return null;
  const submit = recordField(requiredAction, "submit_tool_outputs");
  if (!submit) return null;
  const toolCalls = arrayField(submit, "tool_calls");
  if (!toolCalls.length) return null;
  const first = toolCalls[0] as Record<string, unknown>;
  const callId = stringField(first, "id");
  const fn = recordField(first, "function");
  const toolName = fn ? stringField(fn, "name") : "";
  const argsText = fn ? stringField(fn, "arguments") : "";
  let question: string | undefined;
  let candidates: string[] | undefined;
  const parsed = parseJson(argsText);
  if (parsed && typeof parsed === "object") {
    const argsRecord = parsed as Record<string, unknown>;
    const q = argsRecord.question;
    if (typeof q === "string") {
      question = q;
    }
    const cand = argsRecord.candidates;
    if (Array.isArray(cand)) {
      candidates = cand.filter((c): c is string => typeof c === "string");
    }
  }
  return {
    response_id: responseId,
    call_id: callId,
    tool_name: toolName,
    question,
    candidates,
  };
}

function responseOutputText(data: Record<string, unknown>): string {
  const parts: string[] = [];
  const reportParts: string[] = [];
  for (const item of arrayField(data, "output")) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      continue;
    }
    const record = item as Record<string, unknown>;
    if (stringField(record, "type") !== "message") {
      continue;
    }
    if (isProcessReasoningMessage(record)) {
      continue;
    }
    for (const block of arrayField(record, "content")) {
      if (!block || typeof block !== "object" || Array.isArray(block)) {
        continue;
      }
      const content = block as Record<string, unknown>;
      const contentType = stringField(content, "type");
      if (contentType !== "output_text" && contentType !== "text") {
        continue;
      }
      const text = stringField(content, "text");
      if (text) {
        if (isFinalReportMessage(record)) {
          reportParts.push(text);
        } else {
          parts.push(text);
        }
      }
    }
  }
  return reportParts.length ? reportParts.join("") : parts.join("");
}

function isFinalReportMessage(record: Record<string, unknown>): boolean {
  const metadata = recordField(record, "metadata");
  return Boolean(metadata?.pai_final_report);
}

function isProcessReasoningMessage(record: Record<string, unknown>): boolean {
  const metadata = recordField(record, "metadata");
  return Boolean(metadata?.pai_process_reasoning);
}

// Some upstream providers (e.g. qwen-plus) emit a placeholder `id` like
// `__fake_id__` for every function_call. Without filtering, all tool calls
// alias to the first one in `toolCallByItemId`, so step 4's tool ends up
// rendered inside step 1's card. `call_id` is always unique per tool call, so
// we dedup primarily by call_id and only trust item ids that look real.
const PLACEHOLDER_ITEM_ID_RE = /^(?:__fake_id__|fake_id|placeholder|tmp|temp|undefined|null)$/i;

function isUsableItemId(itemId: string): boolean {
  return Boolean(itemId) && !PLACEHOLDER_ITEM_ID_RE.test(itemId);
}

function registerToolCall(state: ResponsesParseState, item: Record<string, unknown>): ToolCallEntry {
  const rawItemId = stringField(item, "id");
  const callId = stringField(item, "call_id");
  const name = stringField(item, "name") || "tool";
  const itemId = isUsableItemId(rawItemId) ? rawItemId : "";
  const existing = (callId ? state.toolCallByCallId.get(callId) : undefined)
    || (itemId ? state.toolCallByItemId.get(itemId) : undefined);
  if (existing) {
    if (callId && !state.toolCallByCallId.has(callId)) {
      state.toolCallByCallId.set(callId, existing);
    }
    if (itemId && !state.toolCallByItemId.has(itemId)) {
      state.toolCallByItemId.set(itemId, existing);
    }
    return existing;
  }
  state.toolCounter += 1;
  const entry: ToolCallEntry = {
    toolCallId: callId || itemId || `tool-${state.toolCounter}`,
    callId,
    name,
    index: state.toolCounter,
    argumentsText: stringField(item, "arguments") || "",
    status: "pending",
  };
  if (itemId) state.toolCallByItemId.set(itemId, entry);
  if (callId) state.toolCallByCallId.set(callId, entry);
  return entry;
}

const INTERNAL_TOOL_NAMES = new Set(["update_working_checkpoint", "update_todo", "start_long_term_update", "final_report"]);

function isInternalToolName(name: string) {
  return INTERNAL_TOOL_NAMES.has(name);
}

function stringField(record: Record<string, unknown> | null | undefined, key: string): string {
  const v = record?.[key];
  return typeof v === "string" ? v : "";
}

function recordField(record: Record<string, unknown> | null | undefined, key: string): Record<string, unknown> | null {
  const v = record?.[key];
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

function arrayField(record: Record<string, unknown> | null | undefined, key: string): unknown[] {
  const v = record?.[key];
  return Array.isArray(v) ? v : [];
}

function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function errorMessage(data: Record<string, unknown>): string {
  const err = data.error;
  if (err && typeof err === "object" && !Array.isArray(err)) {
    const m = (err as Record<string, unknown>).message;
    if (typeof m === "string") return m;
  }
  if (typeof err === "string") return err;
  return "";
}
