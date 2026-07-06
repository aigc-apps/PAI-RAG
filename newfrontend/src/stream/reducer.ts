import type { AssistantStep, ChatMessage, FileArtifact, ToolUse } from "../types";

type StreamEvent = Record<string, unknown>;

export interface StreamState {
  message: ChatMessage;
  conversationId?: string;
  responseId?: string;
  lastSequenceNumber: number;
}

export function initialStreamState(id: string): StreamState {
  return {
    message: {
      id,
      role: "assistant",
      text: "",
      reasoning: "",
      reasoningStatus: "idle",
      status: "streaming",
      lastSequenceNumber: 0,
      toolCalls: [],
      steps: [],
    },
    lastSequenceNumber: 0,
  };
}

/** The final answer is the trailing text run — text after the last tool call. */
function finalAnswer(steps: AssistantStep[]): string {
  const last = steps[steps.length - 1];
  return last && last.kind === "text" ? last.text : "";
}

function callIdOf(e: Record<string, unknown>): string {
  const item = e.item as { call_id?: string; id?: string } | undefined;
  const raw = (e.call_id as string) || item?.call_id || (e.item_id as string) || item?.id || "";
  return raw.startsWith("fc_") ? raw.slice(3) : raw;
}

// Narrow helper: identity cast kept for symmetry with call sites.
function f(event: StreamEvent): StreamEvent {
  return event;
}

function reduceCore(state: StreamState, event: StreamEvent): StreamState {
  const e = f(event);
  const msg = state.message;

  switch (e.type) {
    case "response.created": {
      const response = e.response as
        | { id?: string; conversation?: { id?: string } | null }
        | undefined;
      const responseId = response?.id;
      const conversationId = response?.conversation?.id;
      return {
        ...state,
        responseId: responseId ?? state.responseId,
        conversationId: conversationId ?? state.conversationId,
        message: {
          ...msg,
          responseId: responseId ?? msg.responseId,
          id: responseId ?? msg.id,
        },
      };
    }

    case "response.output_text.delta": {
      const delta = String(e.delta ?? "");
      if (!delta) return state;
      // Append to the open text run, or start a new one if a tool call closed
      // the previous run. Keep `text` pointed at the trailing run so it always
      // holds the final answer (earlier runs are interstitial narration).
      const steps = (msg.steps ?? []).slice();
      // Resume path: a reconstructed bubble may carry prior `text` with no steps
      // yet. Seed the open run from it so the resumed delta continues the answer
      // instead of replacing it.
      if (steps.length === 0 && msg.text) {
        steps.push({ kind: "text", text: msg.text });
      }
      const last = steps[steps.length - 1];
      if (last && last.kind === "text") {
        steps[steps.length - 1] = { kind: "text", text: last.text + delta };
      } else {
        steps.push({ kind: "text", text: delta });
      }
      return {
        ...state,
        message: { ...msg, steps, text: finalAnswer(steps) },
      };
    }

    case "response.reasoning_summary_text.delta": {
      return {
        ...state,
        message: {
          ...msg,
          reasoning: msg.reasoning + String(e.delta ?? ""),
          reasoningStatus: "streaming",
        },
      };
    }

    case "response.reasoning_summary_part.done":
    case "response.reasoning_summary_text.done": {
      return { ...state, message: { ...msg, reasoningStatus: "done" } };
    }

    case "response.output_item.done": {
      const item = e.item as { type?: string } | undefined;
      if (item?.type === "reasoning" && msg.reasoningStatus !== "idle") {
        return { ...state, message: { ...msg, reasoningStatus: "done" } };
      }
      return state;
    }

    case "response.output_item.added": {
      const item = e.item as { type?: string; name?: string } | undefined;
      if (item?.type === "function_call") {
        const id = callIdOf(e);
        const existing = msg.toolCalls.find((t) => t.id === id);
        if (existing) {
          return {
            ...state,
            message: {
              ...msg,
              toolCalls: msg.toolCalls.map((t) =>
                t.id === id
                  ? {
                      ...t,
                      name: item.name || t.name,
                      status: t.status === "done" ? t.status : "running",
                    }
                  : t
              ),
            },
          };
        }
        // New tool call closes the open text run: prior prose was narration,
        // and the timeline records the tool at its true chronological position.
        const steps: AssistantStep[] = [...(msg.steps ?? []), { kind: "tool", id }];
        return {
          ...state,
          message: {
            ...msg,
            toolCalls: [
              ...msg.toolCalls,
              { id, name: item.name || "", arguments: "", status: "running" } as ToolUse,
            ],
            steps,
            text: finalAnswer(steps),
          },
        };
      }
      return state;
    }
    case "response.function_call_arguments.delta": {
      const id = callIdOf(e);
      return { ...state, message: { ...msg, toolCalls: msg.toolCalls.map(t =>
        t.id === id ? { ...t, arguments: t.arguments + String(e.delta ?? "") } : t) } };
    }
    case "response.function_call_arguments.done": {
      const id = callIdOf(e);
      return { ...state, message: { ...msg, toolCalls: msg.toolCalls.map(t =>
        t.id === id ? { ...t, name: (e.name as string) || t.name,
          arguments: String(e.arguments ?? t.arguments) } : t) } };
    }
    case "response.tool_result": {
      const id = callIdOf(e);
      const ok = e.ok !== false;
      const files = Array.isArray(e.files) ? (e.files as FileArtifact[]) : undefined;
      return { ...state, message: { ...msg, toolCalls: msg.toolCalls.map(t =>
        t.id === id ? { ...t, status: ok ? "done" : "error",
          output: ok ? String(e.output ?? "") : t.output,
          error: ok ? t.error : String(e.output ?? e.error ?? ""),
          files: files && files.length ? files : t.files } : t) } };
    }

    case "response.completed":
    case "response.incomplete": {
      const response = e.response as
        | {
            id?: string;
            conversation?: { id?: string } | null;
            status?: string;
            usage?: {
              input_tokens?: number;
              output_tokens?: number;
              total_tokens?: number;
            } | null;
          }
        | undefined;
      const usage = response?.usage
        ? {
            input: response.usage.input_tokens ?? 0,
            output: response.usage.output_tokens ?? 0,
            total: response.usage.total_tokens ?? 0,
          }
        : msg.usage;
      // response.incomplete carries status "cancelled" (server-side cancel) — and
      // tolerates "incomplete" (length cap, out of scope) by treating it the same.
      const status: ChatMessage["status"] =
        e.type === "response.completed"
          ? "completed"
          : response?.status === "incomplete"
          ? "cancelled"
          : (response?.status as ChatMessage["status"]) ?? "cancelled";
      return {
        ...state,
        responseId: response?.id ?? state.responseId,
        conversationId: response?.conversation?.id ?? state.conversationId,
        message: {
          ...msg,
          status,
          reasoningStatus:
            msg.reasoningStatus === "streaming" ? "done" : msg.reasoningStatus,
          responseId: response?.id ?? msg.responseId,
          usage,
        },
      };
    }

    case "response.failed": {
      const response = e.response as
        | { error?: { message?: string } | null }
        | undefined;
      return {
        ...state,
        message: {
          ...msg,
          status: "failed",
          error: response?.error?.message ?? "request failed",
        },
      };
    }

    default:
      // response.in_progress, output_item.added, content_part.*, *_text.done,
      // function_call* — no effect on the rendered message.
      return state;
  }
}

export function reduceStreamEvent(
  state: StreamState,
  event: StreamEvent
): StreamState {
  const e = f(event);
  const next = reduceCore(state, event);
  const seq =
    typeof e.sequence_number === "number"
      ? Math.max(state.lastSequenceNumber, e.sequence_number as number)
      : state.lastSequenceNumber;
  if (next === state && seq === state.lastSequenceNumber) return state;
  return {
    ...next,
    lastSequenceNumber: seq,
    message: { ...next.message, lastSequenceNumber: seq },
  };
}
