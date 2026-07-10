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

/**
 * One conversation's isolated runtime slice. The map key ({@link ConvRuntime.key})
 * is a client-generated local id that stays stable for the runtime's whole life —
 * it is NOT the server `conversationId` (which only arrives on the first
 * `response.created`). Decoupling the two lets a brand-new chat stream into a
 * stable slice without a mid-stream rekey, and lets a background run keep
 * accumulating into its own slice regardless of which conversation is on screen.
 */
export interface ConvRuntime {
  key: string;
  messages: ChatMessage[];
  /** Whether a LOCAL stream loop is currently folding events into this runtime.
   * Drives the composer's `isStreaming` for the active runtime. Distinct from the
   * assistant *message*'s own `status`, which tracks the server run and is what
   * `resumeIfInterrupted` keys off — a backgrounded run is runtime-idle but its
   * message stays `"streaming"` so a later switch-back resumes it. */
  status: "idle" | "streaming";
  conversationId?: string;
  lastResponseId?: string;
}

/** Stable empty array so `activeRuntime(s)?.messages ?? EMPTY_MESSAGES` keeps
 * referential identity across renders — a fresh `[]` each call would defeat
 * zustand's shallow bail-out and thrash MessageList. */
export const EMPTY_MESSAGES: ChatMessage[] = [];

/** Retained idle, server-backed runtimes are capped and evicted oldest-first;
 * they can always be re-hydrated from the server via `getConversation`. */
const MAX_IDLE_RUNTIMES = 30;

let draftSeq = 0;
function newKey(): string {
  draftSeq += 1;
  return `draft_${draftSeq}_${Math.random().toString(36).slice(2, 8)}`;
}

function freshRuntime(): ConvRuntime {
  return { key: newKey(), messages: [], status: "idle" };
}

/** Drop empty idle drafts (never let them pile up), then LRU-cap idle
 * server-backed runtimes. Streaming runtimes and any with buffered messages are
 * always kept so switching back to them still shows their state. */
function pruneRuntimes(
  runtimes: Record<string, ConvRuntime>,
  keep: string
): Record<string, ConvRuntime> {
  const out: Record<string, ConvRuntime> = {};
  for (const [k, rt] of Object.entries(runtimes)) {
    // Drop empty idle *drafts* only — a runtime with a conversationId is a real
    // server conversation (kept, and bounded by the LRU cap below) even if its
    // messages haven't been loaded.
    if (
      k !== keep &&
      rt.status === "idle" &&
      rt.messages.length === 0 &&
      !rt.conversationId
    ) {
      continue;
    }
    out[k] = rt;
  }
  const evictable = Object.values(out).filter(
    (rt) => rt.key !== keep && rt.status === "idle" && rt.conversationId
  );
  if (evictable.length > MAX_IDLE_RUNTIMES) {
    // Object insertion order == creation order; drop the oldest overflow.
    for (const rt of evictable.slice(0, evictable.length - MAX_IDLE_RUNTIMES)) {
      delete out[rt.key];
    }
  }
  return out;
}

interface ChatState {
  runtimes: Record<string, ConvRuntime>;
  activeKey: string;
  model: string;
  agentId: string;

  // Runtime lifecycle
  newDraft: () => string;
  activate: (key: string) => void;
  activateByConversationId: (id: string) => boolean;
  hydrate: (detail: ConversationDetail) => string;
  dropByConversationId: (id: string) => void;
  dropByKey: (key: string) => void;

  // Keyed message mutators — the caller captures a key so writes always land in
  // the intended conversation's slice, never in whatever is currently displayed.
  appendMessage: (key: string, m: ChatMessage) => void;
  updateLastOf: (key: string, patch: Partial<ChatMessage>) => void;
  setStatusOf: (key: string, status: "idle" | "streaming") => void;
  setAnchorsOf: (
    key: string,
    a: { conversationId?: string; lastResponseId?: string }
  ) => void;
  dropLastTurnOf: (key: string, fromIndex: number, lastResponseId?: string) => void;

  // Global settings (unchanged from the pre-isolation store)
  setModel: (model: string) => void;
  setAgent: (agentId: string) => void;
  reset: () => void;
}

/** The conversation currently on screen. */
export const activeRuntime = (s: ChatState): ConvRuntime | undefined =>
  s.runtimes[s.activeKey];

export const useChatStore = create<ChatState>((set, get) => {
  const initial = freshRuntime();
  return {
    runtimes: { [initial.key]: initial },
    activeKey: initial.key,
    model: "",
    agentId: "",

    newDraft: () => {
      const rt = freshRuntime();
      set((s) => {
        const runtimes = pruneRuntimes(s.runtimes, "");
        runtimes[rt.key] = rt;
        return { runtimes, activeKey: rt.key };
      });
      return rt.key;
    },

    activate: (key) => set((s) => (s.runtimes[key] ? { activeKey: key } : s)),

    activateByConversationId: (id) => {
      const entry = Object.values(get().runtimes).find(
        (rt) => rt.conversationId === id
      );
      if (!entry) return false;
      set({ activeKey: entry.key });
      return true;
    },

    hydrate: (detail) => {
      const messages = normalizeHistoryMessages(detail);
      let key = "";
      set((s) => {
        const existing = Object.values(s.runtimes).find(
          (rt) => rt.conversationId === detail.id
        );
        const base = existing ?? freshRuntime();
        key = base.key;
        const rt: ConvRuntime = {
          ...base,
          messages,
          status: "idle",
          conversationId: detail.id,
          lastResponseId: detail.latest_response_id ?? undefined,
        };
        return { runtimes: { ...s.runtimes, [key]: rt }, activeKey: key };
      });
      return key;
    },

    dropByConversationId: (id) =>
      set((s) => {
        const entry = Object.values(s.runtimes).find(
          (rt) => rt.conversationId === id
        );
        if (!entry) return s;
        const runtimes = { ...s.runtimes };
        delete runtimes[entry.key];
        let activeKey = s.activeKey;
        if (activeKey === entry.key) {
          const rt = freshRuntime();
          runtimes[rt.key] = rt;
          activeKey = rt.key;
        }
        return { runtimes, activeKey };
      }),

    // Drop a runtime by its local key — used to discard an unsent draft (which
    // has no conversationId and so nothing to delete server-side). If it was the
    // active one, fall back to the most recently used remaining runtime (so
    // deleting a just-created draft returns you to the previous conversation),
    // or a fresh draft when none remain.
    dropByKey: (key) =>
      set((s) => {
        if (!s.runtimes[key]) return s;
        const runtimes = { ...s.runtimes };
        delete runtimes[key];
        let activeKey = s.activeKey;
        if (activeKey === key) {
          const remaining = Object.keys(runtimes);
          if (remaining.length) {
            activeKey = remaining[remaining.length - 1];
          } else {
            const rt = freshRuntime();
            runtimes[rt.key] = rt;
            activeKey = rt.key;
          }
        }
        return { runtimes, activeKey };
      }),

    appendMessage: (key, m) =>
      set((s) => {
        const rt = s.runtimes[key];
        if (!rt) return s;
        return {
          runtimes: { ...s.runtimes, [key]: { ...rt, messages: [...rt.messages, m] } },
        };
      }),

    updateLastOf: (key, patch) =>
      set((s) => {
        const rt = s.runtimes[key];
        if (!rt || rt.messages.length === 0) return s;
        const messages = rt.messages.slice();
        messages[messages.length - 1] = { ...messages[messages.length - 1], ...patch };
        return { runtimes: { ...s.runtimes, [key]: { ...rt, messages } } };
      }),

    setStatusOf: (key, status) =>
      set((s) => {
        const rt = s.runtimes[key];
        if (!rt) return s;
        return { runtimes: { ...s.runtimes, [key]: { ...rt, status } } };
      }),

    setAnchorsOf: (key, { conversationId, lastResponseId }) =>
      set((s) => {
        const rt = s.runtimes[key];
        if (!rt) return s;
        return {
          runtimes: {
            ...s.runtimes,
            [key]: {
              ...rt,
              conversationId: conversationId ?? rt.conversationId,
              lastResponseId: lastResponseId ?? rt.lastResponseId,
            },
          },
        };
      }),

    // Regenerate support: drop the last turn (user + assistant from fromIndex on)
    // and rewind the response anchor. Writes lastResponseId verbatim, so it can
    // clear it to undefined when the first turn is removed.
    dropLastTurnOf: (key, fromIndex, lastResponseId) =>
      set((s) => {
        const rt = s.runtimes[key];
        if (!rt) return s;
        return {
          runtimes: {
            ...s.runtimes,
            [key]: { ...rt, messages: rt.messages.slice(0, fromIndex), lastResponseId },
          },
        };
      }),

    setModel: (model) => set({ model }),
    setAgent: (agentId) => set({ agentId }),

    reset: () => {
      const rt = freshRuntime();
      set({ runtimes: { [rt.key]: rt }, activeKey: rt.key });
    },
  };
});
