import type { AgentUpdate, RunCreateResponse, RunStreamEnvelope, SessionDetail, SessionSummary, StreamEvent } from "@/lib/types";

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

export async function regenerateLastAnswer(sessionId: string, signal?: AbortSignal): Promise<RunCreateResponse> {
  const response = await fetch(`/api/sessions/${sessionId}/regenerate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
    signal,
  });
  await assertOk(response);
  return response.json();
}

export async function createRun(sessionId: string | null, text: string, signal?: AbortSignal): Promise<RunCreateResponse> {
  const headers = new Headers({
    "Content-Type": "application/json",
    ...(sessionId ? { "X-Session-Id": sessionId } : {}),
  });
  const response = await fetch("/api/runs", {
    method: "POST",
    headers,
    body: JSON.stringify({ session_id: sessionId, input: text }),
    signal,
  });
  await assertOk(response);
  return response.json();
}

export async function stopRun(runId: string): Promise<void> {
  const response = await fetch(`/api/runs/${runId}/stop`, { method: "POST" });
  await assertOk(response);
}

export async function streamRunEvents(
  runId: string,
  fallbackSessionId: string,
  onChunk: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch(`/api/runs/${runId}/events`, {
    cache: "no-store",
    signal,
  });
  await assertOk(response);

  const returnedSessionId = response.headers.get("X-Session-Id") || fallbackSessionId;
  const reader = response.body?.getReader();
  if (!reader) {
    return returnedSessionId;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  const state: RunEventState = {
    assistantText: "",
    currentReasoningId: "reasoning-1",
    startedReasoning: new Set(),
    toolCounter: 0,
    activeTools: new Map(),
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const event of events) {
      const lines = event.split("\n").filter((line) => line.startsWith("data: "));
      for (const line of lines) {
        const data = line.slice(6);
        if (!data || data === "[DONE]") {
          continue;
        }
        const payload = JSON.parse(data) as RunStreamEnvelope & { sessionId?: string };
        const updates = runEnvelopeToUpdates(payload, state);
        for (const update of updates) {
          onChunk({ sessionId: payload.session_id || payload.sessionId || returnedSessionId, update });
        }
      }
    }
  }

  return returnedSessionId;
}

type RunEventState = {
  assistantText: string;
  currentReasoningId: string;
  startedReasoning: Set<string>;
  toolCounter: number;
  activeTools: Map<string, string[]>;
};

function textContent(text: string) {
  return { type: "text" as const, text };
}

function toolKind(tool: string) {
  if (tool === "code_run") {
    return "execute";
  }
  if (tool === "file_read") {
    return "read";
  }
  if (tool === "file_write" || tool === "file_patch") {
    return "edit";
  }
  if (tool === "ask_user") {
    return "ask";
  }
  return "tool";
}

function rememberTool(state: RunEventState, tool: string, explicitId?: string) {
  const id = explicitId || `tool-${state.toolCounter + 1}`;
  if (!explicitId) {
    state.toolCounter += 1;
  }
  const active = state.activeTools.get(tool) ?? [];
  if (!active.includes(id)) {
    active.push(id);
  }
  state.activeTools.set(tool, active);
  return id;
}

function resolveTool(state: RunEventState, tool: string, explicitId?: string) {
  if (explicitId) {
    return explicitId;
  }
  const active = state.activeTools.get(tool) ?? [];
  const id = active.pop();
  if (active.length) {
    state.activeTools.set(tool, active);
  } else {
    state.activeTools.delete(tool);
  }
  return id ?? rememberTool(state, tool);
}

function isAgentUpdate(value: unknown): value is AgentUpdate {
  return Boolean(value && typeof value === "object" && "sessionUpdate" in value);
}

function reasoningId(payload: RunStreamEnvelope, state: RunEventState) {
  return payload.step_id || state.currentReasoningId || "reasoning-1";
}

function ensureReasoningStarted(id: string, state: RunEventState, title = "Agent step"): AgentUpdate[] {
  if (state.startedReasoning.has(id)) {
    return [];
  }
  state.startedReasoning.add(id);
  state.currentReasoningId = id;
  return [{ sessionUpdate: "thought_start", thoughtId: id, title, status: "in_progress" }];
}

function runEnvelopeToUpdates(payload: RunStreamEnvelope, state: RunEventState): AgentUpdate[] {
  if (payload.update) {
    return [payload.update];
  }
  if (isAgentUpdate(payload.data)) {
    return [payload.data];
  }

  if (payload.event === "message.delta") {
    const delta = payload.delta ?? "";
    if (!delta) {
      return [];
    }
    state.assistantText += delta;
    return [{ sessionUpdate: "agent_message_chunk", content: textContent(delta) }];
  }

  if (payload.event === "reasoning.started") {
    const id = reasoningId(payload, state);
    state.currentReasoningId = id;
    if (state.startedReasoning.has(id)) {
      return [];
    }
    state.startedReasoning.add(id);
    return [{
      sessionUpdate: "thought_start",
      thoughtId: id,
      title: payload.title || "Agent step",
      status: payload.status || "in_progress",
      hidden: payload.hidden,
    }];
  }

  if (payload.event === "reasoning.available") {
    const text = payload.text ?? "";
    if (!text) {
      return [];
    }
    const id = reasoningId(payload, state);
    const updates: AgentUpdate[] = [];
    updates.push(...ensureReasoningStarted(id, state, payload.title || "Agent step"));
    updates.push({ sessionUpdate: "thought_delta", thoughtId: id, content: textContent(text), replace: payload.replace });
    return updates;
  }

  if (payload.event === "reasoning.completed") {
    const id = reasoningId(payload, state);
    const updates = ensureReasoningStarted(id, state, payload.title || "Agent step");
    if (state.currentReasoningId === id) {
      state.currentReasoningId = "";
    }
    updates.push({
      sessionUpdate: "thought_done",
      thoughtId: id,
      status: payload.status || "completed",
      hidden: payload.hidden,
      content: payload.text ? textContent(payload.text) : undefined,
    });
    return updates;
  }

  if (payload.event === "tool.delta") {
    const tool = payload.tool || "tool";
    const id = rememberTool(state, tool, payload.tool_call_id);
    return [{
      sessionUpdate: "tool_call_delta",
      toolCallId: id,
      index: Number(id.match(/-(\d+)$/)?.[1] ?? 0),
      title: payload.preview || tool,
      name: tool,
      kind: payload.kind || toolKind(tool),
      status: payload.status || "in_progress",
      hidden: payload.hidden,
      argumentsDelta: payload.arguments_delta || "",
      argumentsText: payload.arguments_text || "",
    }];
  }

  if (payload.event === "tool.started") {
    const tool = payload.tool || "tool";
    const id = rememberTool(state, tool, payload.tool_call_id);
    return [{
      sessionUpdate: "tool_call",
      toolCallId: id,
      title: payload.preview || tool,
      name: tool,
      kind: payload.kind || toolKind(tool),
      status: payload.status || "in_progress",
      hidden: payload.hidden,
      input: payload.input,
    }];
  }

  if (payload.event === "tool.updated" || payload.event === "tool.completed") {
    const tool = payload.tool || "tool";
    const id = resolveTool(state, tool, payload.tool_call_id);
    const status = payload.status || (payload.error ? "failed" : "completed");
    return [{
      sessionUpdate: "tool_call_update",
      toolCallId: id,
      status,
      content: payload.content ? textContent(payload.content) : undefined,
      data: payload.data,
    }];
  }

  if (payload.event === "ask_user") {
    return [{
      sessionUpdate: "ask_user",
      question: payload.question || "Please provide input.",
      candidates: payload.candidates || [],
    }];
  }

  if (payload.event === "run.completed") {
    const output = payload.output ?? "";
    if (output && !state.assistantText) {
      state.assistantText = output;
      return [{ sessionUpdate: "agent_message_chunk", content: textContent(output) }];
    }
    return [];
  }

  if (payload.event === "run.failed") {
    const message = typeof payload.error === "string" ? payload.error : "Run failed";
    return [{ sessionUpdate: "agent_message_chunk", content: textContent(`**Error:** ${message}`) }];
  }

  return [];
}
