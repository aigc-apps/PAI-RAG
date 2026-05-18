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
}

export interface PendingHitl {
  response_id: string;
  call_id: string;
  tool_name: string;
  question?: string;
  candidates?: string[];
}

export interface SessionDetail {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  status?: string;
  messages: ChatMessage[];
  pending_hitl?: PendingHitl | null;
}

export interface AskUserPayload {
  question: string;
  candidates?: string[];
}

export interface StreamEvent {
  sessionId: string;
  update: AgentUpdate;
}

export interface OfficialSkill {
  name: string;
  description: string;
  trigger: string;
  allowed_tools: string[];
  source: string;
}

export interface EvolvedSkill {
  name: string;
  description: string;
  kind: "sop" | "script";
  source: string;
}

export interface SkillInventory {
  official: OfficialSkill[];
  evolved: EvolvedSkill[];
}

export interface ResponsesSseEvent {
  type: string;
  data: Record<string, unknown>;
}

export interface ResponsesStreamHandlers {
  onEvent?: (event: ResponsesSseEvent) => void;
  onUpdate?: (event: StreamEvent) => void;
  onRequiresAction?: (pending: PendingHitl & { sessionId?: string }) => void;
  onTerminal?: (kind: "completed" | "failed", payload: Record<string, unknown>) => void;
}
