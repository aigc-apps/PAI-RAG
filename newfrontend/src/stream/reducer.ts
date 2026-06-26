import type { ResponseStreamEvent } from "openai/resources/responses/responses";
import type { ChatMessage } from "../types";

export interface StreamState {
  message: ChatMessage;
  conversationId?: string;
  responseId?: string;
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
    },
  };
}

// Narrow helper: read a field off a loosely-typed event without fighting the
// SDK's giant discriminated union in every branch.
function f(event: ResponseStreamEvent): Record<string, unknown> {
  return event as unknown as Record<string, unknown>;
}

export function reduceStreamEvent(
  state: StreamState,
  event: ResponseStreamEvent
): StreamState {
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
      return {
        ...state,
        message: { ...msg, text: msg.text + String(e.delta ?? "") },
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

    case "response.completed": {
      const response = e.response as
        | {
            id?: string;
            conversation?: { id?: string } | null;
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
      return {
        ...state,
        responseId: response?.id ?? state.responseId,
        conversationId: response?.conversation?.id ?? state.conversationId,
        message: {
          ...msg,
          status: "completed",
          reasoningStatus: msg.reasoningStatus === "streaming" ? "done" : msg.reasoningStatus,
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
