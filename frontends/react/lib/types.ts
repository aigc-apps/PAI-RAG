export type Role = "user" | "assistant" | "system";

export type AgentUpdate =
  | {
      sessionUpdate: "agent_message_chunk";
      content: { type: "text"; text: string };
    }
  | {
      sessionUpdate: "thought_start";
      thoughtId: string;
      title?: string;
      status?: "pending" | "in_progress" | "completed" | "failed";
      hidden?: boolean;
      content?: { type: "text"; text: string };
    }
  | {
      sessionUpdate: "thought_delta";
      thoughtId: string;
      content: { type: "text"; text: string };
      replace?: boolean;
    }
  | {
      sessionUpdate: "thought_done";
      thoughtId: string;
      status: "pending" | "in_progress" | "completed" | "failed";
      hidden?: boolean;
      content?: { type: "text"; text: string };
    }
  | {
      sessionUpdate: "thought";
      title?: string;
      content: { type: "text"; text: string };
    }
  | {
      sessionUpdate: "tool_call";
      toolCallId: string;
      title: string;
      name: string;
      kind: string;
      status: "pending" | "in_progress" | "completed" | "failed";
      hidden?: boolean;
      input?: Record<string, unknown>;
    }
  | {
      sessionUpdate: "tool_call_delta";
      toolCallId: string;
      index: number;
      title?: string;
      name?: string;
      nameDelta?: string;
      kind?: string;
      status: "pending" | "in_progress" | "completed" | "failed";
      hidden?: boolean;
      argumentsDelta?: string;
      argumentsText?: string;
    }
  | {
      sessionUpdate: "tool_call_update";
      toolCallId: string;
      status: "pending" | "in_progress" | "completed" | "failed";
      content?: { type: "text"; text: string };
      data?: unknown;
    }
  | {
      sessionUpdate: "ask_user";
      question: string;
      candidates?: string[];
    }
  | {
      sessionUpdate: "done";
      stopReason: string;
    };

export interface ChatMessage {
  role: Role;
  content: string;
  events?: AgentUpdate[];
}

export interface SessionSummary {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count?: number;
  running?: boolean;
  status?: string;
  active_run_id?: string;
}

export interface SessionDetail {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  status?: string;
  active_run_id?: string;
  messages: ChatMessage[];
}

export interface AskUserPayload {
  question: string;
  candidates?: string[];
}

export interface StreamEvent {
  sessionId: string;
  update: AgentUpdate;
}

export interface RunCreateResponse {
  id?: string;
  object?: "agent.run";
  run_id: string;
  session_id?: string;
  status: string;
  stream_from?: string;
  created_at?: number;
}

export interface RunStreamEnvelope {
  event: string;
  run_id: string;
  session_id?: string;
  sequence?: string;
  created_at?: number;
  timestamp?: number;
  output?: string;
  usage?: unknown;
  step_id?: string;
  tool_call_id?: string;
  title?: string;
  status?: "pending" | "in_progress" | "completed" | "failed";
  hidden?: boolean;
  delta?: string;
  text?: string;
  replace?: boolean;
  tool?: string;
  preview?: string;
  kind?: string;
  input?: Record<string, unknown>;
  arguments_delta?: string;
  arguments_text?: string;
  content?: string;
  duration?: number;
  error?: boolean | string;
  question?: string;
  candidates?: string[];
  data?: unknown;
  update?: AgentUpdate;
}

export interface UserProfile {
  user_id: string;
  username: string;
  created_at?: string;
  updated_at?: string;
  is_service?: boolean;
}

export interface AuthResponse {
  access_token: string;
  token_type: "bearer";
  user: UserProfile;
}
