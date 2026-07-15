export type MessageStatus =
  | "streaming"
  | "completed"
  | "failed"
  | "stopped"
  | "cancelled";
export type ReasoningStatus = "idle" | "streaming" | "done";

/**
 * A file the agent surfaced from the sandbox (via publish_artifact). `id` is a
 * signed token fetched from `/v1/files/{id}`; `kind` drives preview-vs-download.
 */
export interface FileArtifact {
  id: string;
  name: string;
  mime: string;
  size: number;
  kind: "image" | "markdown" | "html" | "text" | "file";
}

/**
 * A structured, stream-only notice a tool surfaced beside its text output.
 * Currently only the aliyun authorization card: emitted when an `aliyun` CLI
 * call in the sandbox fails with a credential-class error. `bound` tells the
 * card whether to offer "去授权" (unbound) or "重新校验/重新授权" (bound).
 * Never persisted — a reloaded conversation won't replay it.
 */
export interface AliyunAuthNotice {
  kind: "aliyun_authorization";
  bound: boolean;
  error_code?: string;
  /**
   * Human-in-the-loop marker. When true, the agent turn paused here and handed
   * control to the user — the card is a "waiting for you" affordance, not a
   * passive log entry. Resolving it (authorize / re-verify) offers a "继续"
   * button that resumes the agent.
   */
  interrupt?: boolean;
}

export type ToolNotice = AliyunAuthNotice;

export interface ToolUse {
  id: string;
  name: string;
  arguments: string;
  status: "running" | "done" | "error";
  output?: string;
  error?: string;
  /** Structured file artifacts this tool produced (separate from `output`). */
  files?: FileArtifact[];
  /** Structured UI notice this tool surfaced (e.g. an authorization card). */
  notice?: ToolNotice;
  /** Client clock (ms) when the tool call first appeared; used to derive duration. */
  startedAt?: number;
  /** Elapsed time (ms) from the call appearing to its result, when known. */
  durationMs?: number;
}

/**
 * One entry in an assistant turn's ordered timeline. Reasoning and ordinary
 * assistant prose are separate text runs; a `tool` step references a
 * {@link ToolUse} by id. Steps are appended in stream order so the UI can keep
 * the execution process interleaved and reserve only the final text run for the
 * answer body.
 */
export type AssistantStep =
  | { kind: "reasoning"; text: string }
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
   * Ordered reasoning/text/tool timeline. New history records persist it;
   * pre-change history omits it and falls back to aggregate fields.
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
