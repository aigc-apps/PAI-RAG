export type MessageStatus =
  | "streaming"
  | "completed"
  | "failed"
  | "stopped"
  | "cancelled";
export type ReasoningStatus = "idle" | "streaming" | "done";

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
