export type Role = "user" | "assistant" | "system";

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count?: number;
  running?: boolean;
}

export interface SessionDetail {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
}

export interface AskUserPayload {
  question: string;
  candidates?: string[];
}

export interface StreamEvent {
  sessionId: string;
  content: string;
}
