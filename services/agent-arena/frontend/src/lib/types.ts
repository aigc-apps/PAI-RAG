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
  item_type?: string | null
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
  kind: "arena"
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

export type BatchHistoryAgentSummary = {
  agent_key: "a" | "b"
  agent_name: string
  agent_model: string
  total: number
  success: number
  success_rate: number | null
  latency_p50_ms: number | null
  latency_p90_ms: number | null
}

export type BatchHistorySummary = {
  kind: "batch"
  batch_id: string
  created_at: string
  target: string
  mode: string
  iterations: number
  concurrency: number
  cancelled: boolean
  input: string
  item_count: number
  success_count: number
  success_rate: number | null
  agent_summaries: BatchHistoryAgentSummary[]
  consistency_count: number
  latest_consistency: ConsistencyResponse | null
}

export type HistoryItem = HistorySummary | BatchHistorySummary

export type HistoryListResponse = {
  items: HistoryItem[]
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

export type ConsistencyIssue = {
  index: number
  agent_key: "a" | "b"
  problem: string
  severity: "low" | "medium" | "high"
}

export type ConsistencyAgentReport = {
  agent_key: "a" | "b"
  agent_name: string
  samples_evaluated: number
  stable: boolean
  consistency_score: number
  summary: string
  issues: ConsistencyIssue[]
  suggestions: string[]
}

export type ConsistencyResponse = {
  ok: boolean
  batch_id: string
  created_at: string
  model: string
  latency_ms: number | null
  reports: ConsistencyAgentReport[]
  error: string | null
  raw: Record<string, unknown>
}

export type BatchDetailItem = {
  index: number
  agent_key: "a" | "b"
  agent_name: string
  agent_model: string
  ok: boolean
  latency_ms: number | null
  content: string
  content_length: number
  finish_reason: string | null
  error: string | null
  assertion_passed: boolean | null
  trace_summary: TraceSummary
  trace_events: AgentTraceEvent[]
}

export type BatchDetailResponse = {
  batch_id: string
  created_at: string
  target: string
  mode: string
  iterations: number
  concurrency: number
  cancelled: boolean
  request: Record<string, unknown>
  summaries: Record<string, Record<string, unknown>>
  items: BatchDetailItem[]
  consistency_results: ConsistencyResponse[]
}

export type ViewMode =
  | "arena"
  | "batch"
  | "history"
  | "endpoints"
  | "judge-model"
  | "datasets"

// ---- DB-backed config & dataset entities ----------------------------------

export type EnvVarStatus = {
  name: string
  present: boolean
  preview: string
}

export type AgentDef = {
  id: string
  created_at: string
  updated_at: string
  name: string
  base_url: string
  model: string
  trace_mode: "responses" | "chat" | "runs"
  api_key_env: EnvVarStatus
  runs_base_url: string
  headers: Record<string, string>
  description: string
}

export type ActivePair = {
  a_agent_id: string | null
  b_agent_id: string | null
  updated_at: string | null
}

export type AgentListResponse = {
  agents: AgentDef[]
  active_pair: ActivePair
}

export type JudgeConfigDef = {
  base_url: string
  model: string
  api_key_env: EnvVarStatus
  source: "db" | "env"
  updated_at?: string | null
}

export type Dataset = {
  id: string
  created_at: string
  updated_at: string
  name: string
  description: string
  case_count?: number
  last_run?: DatasetRun | null
}

export type DatasetCase = {
  id: string
  dataset_id: string
  created_at: string
  updated_at: string
  query: string
  system_prompt: string
  expected_answer: string
  tags: string[]
  source_run_id: string | null
}

export type DatasetWithChildren = Dataset & {
  cases: DatasetCase[]
  runs: DatasetRun[]
}

export type DatasetRunWinners = {
  a: number
  b: number
  tie: number
  unknown: number
}

export type DatasetRunSummary = {
  total?: number
  completed?: number
  failed?: number
  judged?: number
  winners?: DatasetRunWinners
}

export type DatasetRun = {
  id: string
  dataset_id: string
  created_at: string
  finished_at: string | null
  status: "running" | "completed" | "completed_with_errors" | "failed" | string
  agent_a_id: string | null
  agent_b_id: string | null
  judge_model: string | null
  summary: DatasetRunSummary
}

export type DatasetRunItemBody = {
  case?: { id: string; query: string; expected_answer: string }
  agents?: {
    a?: { name?: string; model?: string; ok?: boolean; latency_ms?: number | null; error?: string | null; content_preview?: string }
    b?: { name?: string; model?: string; ok?: boolean; latency_ms?: number | null; error?: string | null; content_preview?: string }
  }
  judge?: {
    ok?: boolean
    winner?: string | null
    summary?: string
    error?: string | null
  }
  error?: string
}

export type DatasetRunItem = {
  id: string
  run_id: string
  case_id: string
  idx: number
  status: string
  a_run_id: string | null
  b_run_id: string | null
  judge_result_id: string | null
  body: DatasetRunItemBody
}

export type DatasetRunDetail = DatasetRun & {
  items: DatasetRunItem[]
}

export type ArenaRunCandidate = {
  run_id: string
  created_at: string
  input: string
  system: string
  agent_a_name: string
  agent_b_name: string
}
