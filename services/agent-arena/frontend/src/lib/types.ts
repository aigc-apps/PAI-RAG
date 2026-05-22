export type PublicAgentConfig = {
  name: string
  base_url: string
  model: string
  configured: boolean
  has_api_key: boolean
  completion_path: string
  trace_mode: "chat" | "runs"
  runs_path: string
}

export type PublicJudgeConfig = {
  base_url: string
  model: string
  configured: boolean
  has_api_key: boolean
  completion_path: string
}

export type ConfigResponse = {
  agents: {
    a: PublicAgentConfig
    b: PublicAgentConfig
  }
  judge: PublicJudgeConfig
  timeout_seconds: number
  judge_timeout_seconds: number
  auth_required?: boolean
}

export type AgentTraceEvent = {
  event: string
  timestamp?: number | null
  tool?: string | null
  preview?: string | null
  duration?: number | null
  error?: boolean | string | null
  text?: string | null
  delta?: string | null
}

export type TraceSummary = {
  supported?: boolean
  event_count?: number
  tool_call_count?: number
  failed_tool_count?: number
  total_tool_duration_s?: number
  reasoning_count?: number
  message_delta_count?: number
  completion_event?: string | null
}

export type AgentResult = {
  ok: boolean
  name: string
  model: string
  content: string
  latency_ms: number | null
  error: string | null
  raw_finish_reason: string | null
  trace_supported: boolean
  trace_events: AgentTraceEvent[]
  trace_summary: TraceSummary
}

export type CompareResponse = {
  run_id: string
  request: Record<string, unknown>
  agents: {
    a: AgentResult
    b: AgentResult
  }
}

export type JudgeResponse = {
  ok: boolean
  run_id?: string | null
  judge_id?: string | null
  winner: "a" | "b" | "tie" | string | null
  summary: string
  answer_scores: Record<string, unknown>
  process_scores: Record<string, unknown>
  strengths: Record<string, string[]>
  weaknesses: Record<string, string[]>
  recommendations: Record<string, string[]>
  latency_ms: number | null
  error: string | null
  raw: Record<string, unknown>
}

export type HistoryJudgeRecord = {
  judge_id: string
  run_id: string
  created_at: string
  model: string
  result: JudgeResponse
}

export type HistorySummary = {
  run_id: string
  created_at: string
  updated_at: string
  input: string
  system: string
  temperature: number | null
  max_tokens: number | null
  agent_a: Record<string, unknown>
  agent_b: Record<string, unknown>
  judge_count: number
  latest_judge: HistoryJudgeRecord | null
}

export type HistoryListResponse = {
  items: HistorySummary[]
  total: number
  limit: number
  offset: number
}

export type HistoryDetailResponse = {
  run_id: string
  created_at: string
  updated_at: string
  input: string
  system: string
  temperature: number | null
  max_tokens: number | null
  compare: CompareResponse
  judges: HistoryJudgeRecord[]
}

export type ViewMode = "arena" | "batch" | "history"
