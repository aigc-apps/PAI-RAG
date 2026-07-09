import { create } from "zustand";
import type { ChatMessage, ConversationDetail, FileArtifact, ToolNotice } from "../types";

interface WireHistoryMessage {
  role: "user" | "assistant";
  text: string;
  reasoning?: string;
  response_id: string;
  previous_response_id?: string | null;
  status?: ChatMessage["status"];
  tool_calls?: Array<{
    call_id: string;
    name: string;
    arguments?: string;
    output?: string;
    files?: FileArtifact[];
    notice?: ToolNotice;
  }>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  } | null;
}

export function normalizeHistoryMessages(
  detail: ConversationDetail
): ChatMessage[] {
  const rows = detail.messages as unknown as WireHistoryMessage[];
  return rows.map((r, i) => {
    const reasoning = r.reasoning ?? "";
    return {
      id: r.role === "assistant" ? r.response_id : `${r.response_id}:u:${i}`,
      role: r.role,
      text: r.text ?? "",
      reasoning,
      reasoningStatus: reasoning ? "done" : "idle",
      status: r.status ?? "completed",
      responseId: r.response_id,
      previousResponseId: r.previous_response_id ?? undefined,
      toolCalls: (r.tool_calls ?? []).map((tc) => ({
        id: tc.call_id, name: tc.name, arguments: tc.arguments ?? "",
        status: "done" as const, output: tc.output ?? "",
        files: tc.files && tc.files.length ? tc.files : undefined,
        notice: tc.notice ?? undefined,
      })),
      usage: r.usage
        ? {
            input: r.usage.input_tokens ?? 0,
            output: r.usage.output_tokens ?? 0,
            total: r.usage.total_tokens ?? 0,
          }
        : undefined,
    };
  });
}

interface ChatState {
  messages: ChatMessage[];
  status: "idle" | "streaming";
  model: string;
  agentId: string;
  conversationId?: string;
  lastResponseId?: string;
  appendMessage: (m: ChatMessage) => void;
  updateLast: (patch: Partial<ChatMessage>) => void;
  setStatus: (status: "idle" | "streaming") => void;
  setModel: (model: string) => void;
  setAgent: (agentId: string) => void;
  setAnchors: (a: { conversationId?: string; lastResponseId?: string }) => void;
  dropLastTurn: (fromIndex: number, lastResponseId?: string) => void;
  loadHistory: (detail: ConversationDetail) => void;
  reset: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  status: "idle",
  model: "",
  agentId: "",
  conversationId: undefined,
  lastResponseId: undefined,

  appendMessage: (m) => set((s) => ({ messages: [...s.messages, m] })),

  updateLast: (patch) =>
    set((s) => {
      if (s.messages.length === 0) return s;
      const messages = s.messages.slice();
      messages[messages.length - 1] = {
        ...messages[messages.length - 1],
        ...patch,
      };
      return { messages };
    }),

  setStatus: (status) => set({ status }),
  setModel: (model) => set({ model }),
  setAgent: (agentId) => set({ agentId }),
  setAnchors: ({ conversationId, lastResponseId }) =>
    set((s) => ({
      conversationId: conversationId ?? s.conversationId,
      lastResponseId: lastResponseId ?? s.lastResponseId,
    })),

  // Regenerate support: drop the last turn (user + assistant from fromIndex on)
  // and rewind the response anchor. Unlike setAnchors this WRITES the anchor
  // verbatim, so it can clear it to undefined when the first turn is removed.
  dropLastTurn: (fromIndex, lastResponseId) =>
    set((s) => ({
      messages: s.messages.slice(0, fromIndex),
      lastResponseId,
    })),

  loadHistory: (detail) =>
    set({
      messages: normalizeHistoryMessages(detail),
      conversationId: detail.id,
      lastResponseId: detail.latest_response_id ?? undefined,
      status: "idle",
    }),

  reset: () =>
    set({
      messages: [],
      status: "idle",
      conversationId: undefined,
      lastResponseId: undefined,
    }),
}));
