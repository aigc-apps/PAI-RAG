export type MessageStatus =
  | "streaming"
  | "completed"
  | "failed"
  | "stopped"
  | "cancelled";
export type ReasoningStatus = "idle" | "streaming" | "done";

export interface ToolUse {
  id: string;
  name: string;
  arguments: string;
  status: "running" | "done" | "error";
  output?: string;
  error?: string;
}

/**
 * One entry in an assistant turn's ordered timeline. A `text` step is a run of
 * assistant prose; a `tool` step references a {@link ToolUse} by id. Steps are
 * appended in stream order, so interstitial narration ("我先加载技能…") and the
 * tool calls it precedes stay interleaved — letting the UI show that narration
 * as working/thinking and keep only the final text run as the answer body.
 */
export type AssistantStep =
  | { kind: "text"; text: string }
  | { kind: "tool"; id: string };

export interface ChatMessage {
  /** local list key; equals responseId for assistant turns once known */
  id: string;
  role: "user" | "assistant";
  text: string;
  reasoning: string;
  reasoningStatus: ReasoningStatus;
  status: MessageStatus;
  responseId?: string;
  previousResponseId?: string;
  error?: string;
  usage?: { input: number; output: number; total: number };
  /** highest response.* sequence_number folded so far (resume cursor) */
  lastSequenceNumber?: number;
  toolCalls: ToolUse[];
  /**
   * Ordered text/tool timeline for a live-streamed assistant turn. Absent on
   * reloaded history (persistence collapses the turn to one `text` blob), where
   * the UI falls back to `text` + `toolCalls`.
   */
  steps?: AssistantStep[];
}

export interface ConversationSummary {
  id: string;
  title: string | null;
  created_at: string | null;
  updated_at: string | null;
  last_response_id: string | null;
}

export interface ConversationDetail {
  id: string;
  title: string | null;
  created_at: string | null;
  updated_at: string | null;
  latest_response_id: string | null;
  messages: ChatMessage[];
}
