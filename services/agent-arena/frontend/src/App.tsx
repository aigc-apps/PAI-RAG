import { FormEvent, useEffect, useMemo, useState } from "react"
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Eraser,
  FlaskConical,
  Gavel,
  History,
  KeyRound,
  ListRestart,
  Loader2,
  LogOut,
  RefreshCcw,
  Scale,
  SendHorizontal,
  Server,
  Settings2,
  Swords,
  Wrench,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import {
  apiFetch,
  clearArenaAccessKey,
  getArenaAccessKey,
  isUnauthorizedError,
  readApiJson,
  setArenaAccessKey,
} from "@/lib/api"
import { normalizeError, reportFrontendLog } from "@/lib/frontendLogger"
import { cn } from "@/lib/utils"
import { BatchPage } from "@/BatchPage"

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

type AgentTraceEvent = {
  event: string
  timestamp?: number | null
  tool?: string | null
  preview?: string | null
  duration?: number | null
  error?: boolean | string | null
  text?: string | null
  delta?: string | null
}

type TraceSummary = {
  supported?: boolean
  event_count?: number
  tool_call_count?: number
  failed_tool_count?: number
  total_tool_duration_s?: number
  reasoning_count?: number
  message_delta_count?: number
  completion_event?: string | null
}

type AgentResult = {
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

type CompareResponse = {
  run_id: string
  request: Record<string, unknown>
  agents: {
    a: AgentResult
    b: AgentResult
  }
}

type JudgeResponse = {
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

type HistoryJudgeRecord = {
  judge_id: string
  run_id: string
  created_at: string
  model: string
  result: JudgeResponse
}

type HistorySummary = {
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

type HistoryListResponse = {
  items: HistorySummary[]
  total: number
  limit: number
  offset: number
}

type HistoryDetailResponse = {
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

type ViewMode = "arena" | "batch" | "history"

const samplePrompt =
  "请用三个要点说明：如果要给一个项目加入长期记忆能力，最重要的设计取舍是什么？"

function formatLatency(value: number | null | undefined) {
  if (value === null || value === undefined) return "未返回"
  if (value < 1000) return `${value} ms`
  return `${(value / 1000).toFixed(1)} s`
}

function formatDateTime(value: string | null | undefined) {
  if (!value) return "未知时间"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString("zh-CN", { hour12: false })
}

function modelLabel(agent?: PublicAgentConfig) {
  if (!agent) return "未加载配置"
  return agent.model || "未配置模型"
}

function AgentStatusBadge({ agent }: { agent?: PublicAgentConfig }) {
  if (!agent) return <Badge variant="secondary">加载中</Badge>
  if (!agent.configured) return <Badge variant="warning">未配置</Badge>
  return (
    <Badge variant={agent.has_api_key ? "success" : "secondary"}>
      {agent.has_api_key ? "已配置 key" : "无 key"}
    </Badge>
  )
}

function ResultBadge({ result, loading }: { result?: AgentResult; loading: boolean }) {
  if (loading) {
    return (
      <Badge variant="secondary" className="gap-1">
        <Loader2 className="size-3 animate-spin" />
        请求中
      </Badge>
    )
  }
  if (!result) return <Badge variant="outline">待提交</Badge>
  return result.ok ? (
    <Badge variant="success" className="gap-1">
      <CheckCircle2 className="size-3" />
      成功
    </Badge>
  ) : (
    <Badge variant="destructive" className="gap-1">
      <AlertTriangle className="size-3" />
      失败
    </Badge>
  )
}

function AgentResultPanel({
  title,
  config,
  result,
  loading,
}: {
  title: string
  config?: PublicAgentConfig
  result?: AgentResult
  loading: boolean
}) {
  const summary = result?.trace_summary || {}
  const displayName = config?.name || result?.name || title
  const displayModel = config ? modelLabel(config) : result?.model || "未加载配置"
  const traceModeText = config?.trace_mode === "runs" || result?.trace_supported ? "runs trace" : "chat only"
  return (
    <Card className="min-h-[520px] overflow-hidden">
      <CardHeader className="space-y-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <CardTitle className="flex items-center gap-2">
              <Server className="size-4 text-primary" />
              <span className="truncate">{displayName}</span>
            </CardTitle>
            <CardDescription className="mt-1 truncate">
              {displayModel}
            </CardDescription>
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-2">
            <Badge variant="outline">{traceModeText}</Badge>
            {config ? <AgentStatusBadge agent={config} /> : null}
            <ResultBadge result={result} loading={loading} />
          </div>
        </div>
        <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-2">
          <div className="truncate">
            Endpoint: {config?.completion_path || "历史记录"}
          </div>
          <div className="flex items-center gap-1 sm:justify-end">
            <Clock3 className="size-3.5" />
            {formatLatency(result?.latency_ms)}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
          <Metric label="工具调用" value={summary.tool_call_count ?? 0} />
          <Metric label="失败工具" value={summary.failed_tool_count ?? 0} />
          <Metric label="工具耗时" value={`${summary.total_tool_duration_s ?? 0}s`} />
          <Metric label="过程事件" value={summary.event_count ?? 0} />
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="answer">
          <TabsList>
            <TabsTrigger value="answer">答案</TabsTrigger>
            <TabsTrigger value="trace">过程</TabsTrigger>
          </TabsList>
          <TabsContent value="answer">
            {loading ? (
              <div className="space-y-3">
                <Skeleton className="h-4 w-11/12" />
                <Skeleton className="h-4 w-10/12" />
                <Skeleton className="h-4 w-9/12" />
                <Skeleton className="h-40 w-full" />
              </div>
            ) : result?.error ? (
              <Alert variant="destructive">
                <AlertTriangle className="size-4" />
                <AlertTitle>请求失败</AlertTitle>
                <AlertDescription>
                  <pre className="mt-2 max-h-[280px] whitespace-pre-wrap break-words text-xs">
                    {result.error}
                  </pre>
                </AlertDescription>
              </Alert>
            ) : result?.content ? (
              <ScrollArea className="h-[300px] rounded-md border bg-muted/30 p-4">
                <pre className="whitespace-pre-wrap break-words text-sm leading-6">
                  {result.content}
                </pre>
              </ScrollArea>
            ) : (
              <EmptyBox text="等待输入并提交对比" />
            )}
          </TabsContent>
          <TabsContent value="trace">
            <TraceTimeline result={result} loading={loading} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border bg-muted/30 px-3 py-2">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="mt-1 truncate text-sm font-semibold">{value}</div>
    </div>
  )
}

function EmptyBox({ text }: { text: string }) {
  return (
    <div className="flex h-[300px] items-center justify-center rounded-md border border-dashed bg-muted/20 text-sm text-muted-foreground">
      {text}
    </div>
  )
}

function TraceTimeline({ result, loading }: { result?: AgentResult; loading: boolean }) {
  if (loading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-10/12" />
      </div>
    )
  }
  if (!result) return <EmptyBox text="等待过程事件" />
  if (!result.trace_supported) {
    return (
      <Alert>
        <Activity className="size-4" />
        <AlertTitle>未提供过程事件</AlertTitle>
        <AlertDescription>
          当前 Agent 未启用过程事件。PAI-RAG 可使用 `TRACE_MODE=responses`，外部 Hermes 风格 Agent 可使用 `TRACE_MODE=runs`。
        </AlertDescription>
      </Alert>
    )
  }
  if (!result.trace_events.length) {
    return <EmptyBox text="已启用 runs trace，但没有收到过程事件" />
  }
  return (
    <ScrollArea className="h-[300px] rounded-md border bg-muted/20 p-4">
      <div className="space-y-3">
        {result.trace_events.map((event, index) => (
          <TraceEventRow event={event} key={`${event.event}-${index}`} />
        ))}
      </div>
    </ScrollArea>
  )
}

function TraceEventRow({ event }: { event: AgentTraceEvent }) {
  const isTool = event.event.startsWith("tool.")
  const isError = Boolean(event.error) || event.event === "run.failed"
  return (
    <div className="rounded-md border bg-background p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={isError ? "destructive" : isTool ? "secondary" : "outline"} className="gap-1">
          {isTool ? <Wrench className="size-3" /> : <Activity className="size-3" />}
          {event.event}
        </Badge>
        {event.tool ? <span className="text-sm font-medium">{event.tool}</span> : null}
        {event.duration !== null && event.duration !== undefined ? (
          <span className="text-xs text-muted-foreground">{event.duration}s</span>
        ) : null}
      </div>
      {event.preview || event.text || event.delta ? (
        <pre className="mt-2 whitespace-pre-wrap break-words text-xs leading-5 text-muted-foreground">
          {event.preview || event.text || event.delta}
        </pre>
      ) : null}
    </div>
  )
}

function JudgePanel({
  judge,
  judgeLoading,
  canJudge,
  config,
  onJudge,
  showAction = true,
  agentNames,
}: {
  judge: JudgeResponse | null
  judgeLoading: boolean
  canJudge: boolean
  config: ConfigResponse | null
  onJudge: () => void
  showAction?: boolean
  agentNames?: { a?: string; b?: string }
}) {
  const winnerText =
    judge?.winner === "a"
      ? agentNames?.a || config?.agents.a.name || "Agent A"
      : judge?.winner === "b"
        ? agentNames?.b || config?.agents.b.name || "Agent B"
        : judge?.winner === "tie"
          ? "平局"
          : "未判定"
  return (
    <Card>
      <CardHeader className="space-y-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Gavel className="size-4 text-primary" />
              Judge 裁判员
            </CardTitle>
            <CardDescription className="mt-1">
              使用 OpenAI Judge 模型评估最终答案和公开过程事件。
            </CardDescription>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant={config?.judge.configured ? "success" : "warning"}>
              {config?.judge.configured ? "Judge 已配置" : "Judge 未配置"}
            </Badge>
            <Badge variant="outline">{config?.judge.model || "未配置模型"}</Badge>
            {showAction ? (
              <Button type="button" onClick={onJudge} disabled={!canJudge || judgeLoading}>
                {judgeLoading ? <Loader2 className="size-4 animate-spin" /> : <Scale className="size-4" />}
                对比
              </Button>
            ) : null}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {judgeLoading ? (
          <div className="space-y-3">
            <Skeleton className="h-5 w-40" />
            <Skeleton className="h-24 w-full" />
          </div>
        ) : judge?.error ? (
          <Alert variant="destructive">
            <AlertTriangle className="size-4" />
            <AlertTitle>Judge 失败</AlertTitle>
            <AlertDescription>{judge.error}</AlertDescription>
          </Alert>
        ) : judge?.ok ? (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={judge.winner === "tie" ? "secondary" : "success"}>胜者：{winnerText}</Badge>
              <Badge variant="outline">耗时 {formatLatency(judge.latency_ms)}</Badge>
            </div>
            <p className="text-sm leading-6 text-muted-foreground">{judge.summary}</p>
            <div className="grid gap-4 xl:grid-cols-2">
              <ScoreCard title="最终答案评分" value={judge.answer_scores} />
              <ScoreCard title="过程评分" value={judge.process_scores} />
            </div>
            <div className="grid gap-4 xl:grid-cols-3">
              <ListCard title="优势" data={judge.strengths} />
              <ListCard title="短板" data={judge.weaknesses} />
              <ListCard title="改进建议" data={judge.recommendations} />
            </div>
          </div>
        ) : (
          <EmptyBox text={showAction ? "生成两个答案后点击“对比”开始裁判评估" : "这次 PK 还没有裁判评估"} />
        )}
      </CardContent>
    </Card>
  )
}

function ScoreCard({ title, value }: { title: string; value: Record<string, unknown> }) {
  return (
    <div className="rounded-md border p-4">
      <div className="mb-3 text-sm font-semibold">{title}</div>
      <CodeBlock compact value={JSON.stringify(value || {}, null, 2)} />
    </div>
  )
}

function ListCard({ title, data }: { title: string; data: Record<string, string[]> }) {
  return (
    <div className="rounded-md border p-4">
      <div className="mb-3 text-sm font-semibold">{title}</div>
      <div className="space-y-3 text-sm">
        {(["a", "b"] as const).map((key) => (
          <div key={key}>
            <div className="mb-1 text-xs font-medium uppercase text-muted-foreground">Agent {key.toUpperCase()}</div>
            <ul className="list-disc space-y-1 pl-4">
              {(data?.[key] || ["暂无"]).map((item, index) => (
                <li key={`${key}-${index}`}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  )
}

function historyAgentName(summary: Record<string, unknown>, fallback: string) {
  return typeof summary.name === "string" && summary.name ? summary.name : fallback
}

function historyAgentModel(summary: Record<string, unknown>) {
  return typeof summary.model === "string" && summary.model ? summary.model : "未记录模型"
}

function historyAgentOk(summary: Record<string, unknown>) {
  return summary.ok === true
}

function winnerTextFromJudge(judge: JudgeResponse | null | undefined, item: HistorySummary) {
  if (!judge?.winner) return "未评估"
  if (judge.winner === "tie") return "平局"
  if (judge.winner === "a") return historyAgentName(item.agent_a, "Agent A")
  if (judge.winner === "b") return historyAgentName(item.agent_b, "Agent B")
  return "未判定"
}

function HistoryPage({
  items,
  detail,
  selectedRunId,
  loading,
  detailLoading,
  error,
  config,
  onRefresh,
  onSelect,
}: {
  items: HistorySummary[]
  detail: HistoryDetailResponse | null
  selectedRunId: string
  loading: boolean
  detailLoading: boolean
  error: string
  config: ConfigResponse | null
  onRefresh: () => void
  onSelect: (runId: string) => void
}) {
  return (
    <section className="grid gap-5 xl:grid-cols-[420px_1fr]">
      <Card className="h-fit">
        <CardHeader className="space-y-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-2">
                <History className="size-4 text-primary" />
                历史对比
              </CardTitle>
              <CardDescription className="mt-1">
                每次 PK 和裁判评估都会保存在 SQLite。
              </CardDescription>
            </div>
            <Button type="button" variant="outline" size="sm" onClick={onRefresh} disabled={loading}>
              {loading ? <Loader2 className="size-4 animate-spin" /> : <RefreshCcw className="size-4" />}
              刷新
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {error ? (
            <Alert variant="destructive" className="mb-4">
              <AlertTriangle className="size-4" />
              <AlertTitle>历史读取失败</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}
          {loading ? (
            <div className="space-y-3">
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
            </div>
          ) : items.length ? (
            <div className="space-y-3">
              {items.map((item) => (
                <HistoryListItem
                  item={item}
                  selected={item.run_id === selectedRunId}
                  onClick={() => onSelect(item.run_id)}
                  key={item.run_id}
                />
              ))}
            </div>
          ) : (
            <EmptyBox text="还没有历史记录" />
          )}
        </CardContent>
      </Card>

      <div className="space-y-5">
        {detailLoading ? (
          <div className="space-y-5">
            <Skeleton className="h-32 w-full" />
            <div className="grid gap-5 lg:grid-cols-2">
              <Skeleton className="h-[520px] w-full" />
              <Skeleton className="h-[520px] w-full" />
            </div>
          </div>
        ) : detail ? (
          <HistoryDetail detail={detail} config={config} />
        ) : (
          <Card>
            <CardContent className="pt-6">
              <EmptyBox text="从左侧选择一条历史记录查看详情" />
            </CardContent>
          </Card>
        )}
      </div>
    </section>
  )
}

function HistoryListItem({
  item,
  selected,
  onClick,
}: {
  item: HistorySummary
  selected: boolean
  onClick: () => void
}) {
  const latestJudge = item.latest_judge?.result
  const agentAOk = historyAgentOk(item.agent_a)
  const agentBOk = historyAgentOk(item.agent_b)
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "w-full rounded-md border bg-background p-3 text-left transition hover:border-primary/60 hover:bg-muted/40",
        selected && "border-primary bg-primary/5",
      )}
    >
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={latestJudge?.winner && latestJudge.winner !== "tie" ? "success" : "secondary"}>
          {winnerTextFromJudge(latestJudge, item)}
        </Badge>
        <Badge variant="outline">{item.judge_count} 次评估</Badge>
        <span className="ml-auto text-xs text-muted-foreground">{formatDateTime(item.created_at)}</span>
      </div>
      <div className="mt-2 line-clamp-3 text-sm leading-6">{item.input}</div>
      <div className="mt-3 grid gap-2 text-xs text-muted-foreground">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate">{historyAgentName(item.agent_a, "Agent A")}</span>
          <Badge variant={agentAOk ? "success" : "destructive"}>{agentAOk ? "成功" : "失败"}</Badge>
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="truncate">{historyAgentName(item.agent_b, "Agent B")}</span>
          <Badge variant={agentBOk ? "success" : "destructive"}>{agentBOk ? "成功" : "失败"}</Badge>
        </div>
      </div>
    </button>
  )
}

function HistoryDetail({ detail, config }: { detail: HistoryDetailResponse; config: ConfigResponse | null }) {
  const latestJudge = detail.judges[0]?.result ?? null
  const agentNames = {
    a: detail.compare.agents.a.name,
    b: detail.compare.agents.b.name,
  }
  return (
    <div className="space-y-5">
      <div className="rounded-lg border bg-background p-5 shadow-sm">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">Run {detail.run_id}</Badge>
              <Badge variant="secondary">{formatDateTime(detail.created_at)}</Badge>
              <Badge variant="outline">{detail.judges.length} 次裁判评估</Badge>
            </div>
            <h2 className="mt-3 text-lg font-semibold tracking-normal">历史 PK 详情</h2>
            <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-6 text-muted-foreground">
              {detail.input}
            </p>
          </div>
          <div className="grid shrink-0 grid-cols-2 gap-2 text-xs sm:min-w-[260px]">
            <Metric label="Temperature" value={detail.temperature ?? "未记录"} />
            <Metric label="Max tokens" value={detail.max_tokens ?? "未记录"} />
          </div>
        </div>
        {detail.system ? (
          <div className="mt-4 rounded-md border bg-muted/30 p-3">
            <div className="mb-1 text-xs font-medium text-muted-foreground">System prompt</div>
            <pre className="whitespace-pre-wrap break-words text-xs leading-5">{detail.system}</pre>
          </div>
        ) : null}
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <AgentResultPanel title="Agent A" result={detail.compare.agents.a} loading={false} />
        <AgentResultPanel title="Agent B" result={detail.compare.agents.b} loading={false} />
      </div>

      <JudgePanel
        judge={latestJudge}
        judgeLoading={false}
        canJudge={false}
        config={config}
        onJudge={() => undefined}
        showAction={false}
        agentNames={agentNames}
      />

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <ListRestart className="size-4 text-primary" />
            历史原始数据
          </CardTitle>
          <CardDescription>查看保存的 Compare 和 Judge JSON。</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="compare">
            <TabsList>
              <TabsTrigger value="compare">Compare</TabsTrigger>
              <TabsTrigger value="judges">Judges</TabsTrigger>
            </TabsList>
            <Separator className="my-3" />
            <TabsContent value="compare">
              <CodeBlock value={JSON.stringify(detail.compare, null, 2)} />
            </TabsContent>
            <TabsContent value="judges">
              <CodeBlock value={detail.judges.length ? JSON.stringify(detail.judges, null, 2) : "尚无 Judge 结果"} />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}

function App() {
  const [view, setView] = useState<ViewMode>("arena")
  const [config, setConfig] = useState<ConfigResponse | null>(null)
  const [configError, setConfigError] = useState("")
  const [input, setInput] = useState(samplePrompt)
  const [system, setSystem] = useState("你是一个严谨、直接的技术助手。")
  const [temperature, setTemperature] = useState("0.2")
  const [maxTokens, setMaxTokens] = useState("1200")
  const [loading, setLoading] = useState(false)
  const [judgeLoading, setJudgeLoading] = useState(false)
  const [result, setResult] = useState<CompareResponse | null>(null)
  const [judge, setJudge] = useState<JudgeResponse | null>(null)
  const [error, setError] = useState("")
  const [historyItems, setHistoryItems] = useState<HistorySummary[]>([])
  const [historyDetail, setHistoryDetail] = useState<HistoryDetailResponse | null>(null)
  const [selectedHistoryRunId, setSelectedHistoryRunId] = useState("")
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyDetailLoading, setHistoryDetailLoading] = useState(false)
  const [historyError, setHistoryError] = useState("")
  const [accessKeyInput, setAccessKeyInput] = useState(() => getArenaAccessKey())
  const [needsAccessKey, setNeedsAccessKey] = useState(false)

  function markUnauthorized(message = "请输入 AgentArena access key。") {
    setNeedsAccessKey(true)
    setConfigError(message)
  }

  async function loadConfig() {
    setConfigError("")
    try {
      const response = await apiFetch("/config")
      setConfig(await readApiJson<ConfigResponse>(response))
      setNeedsAccessKey(false)
      setAccessKeyInput(getArenaAccessKey())
    } catch (err) {
      if (isUnauthorizedError(err)) {
        markUnauthorized()
        return
      }
      const normalized = normalizeError(err)
      reportFrontendLog({
        level: "error",
        source: "app.load_config",
        message: normalized.message,
        stack: normalized.stack,
      })
      setConfigError(normalized.message)
    }
  }

  function saveAccessKey(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setArenaAccessKey(accessKeyInput)
    setNeedsAccessKey(false)
    void loadConfig()
  }

  function openAccessKeyPrompt() {
    setAccessKeyInput(getArenaAccessKey())
    setNeedsAccessKey(true)
    setConfigError("")
  }

  function resetAccessKey() {
    clearArenaAccessKey()
    setAccessKeyInput("")
    setNeedsAccessKey(true)
    setConfig(null)
    setConfigError("请输入 AgentArena access key。")
  }

  useEffect(() => {
    void loadConfig()
  }, [])

  useEffect(() => {
    if (view === "history") {
      void loadHistory()
    }
  }, [view])

  const parsedTemperature = useMemo(() => {
    const value = Number.parseFloat(temperature)
    if (!Number.isFinite(value)) return 0.2
    return Math.min(2, Math.max(0, value))
  }, [temperature])

  const parsedMaxTokens = useMemo(() => {
    const value = Number.parseInt(maxTokens, 10)
    if (!Number.isFinite(value) || value <= 0) return undefined
    return value
  }, [maxTokens])

  const canJudge = Boolean(
    result &&
      !loading &&
      (result.agents.a.content.trim() || result.agents.b.content.trim()),
  )

  async function loadHistory() {
    setHistoryLoading(true)
    setHistoryError("")
    try {
      const response = await apiFetch("/history?limit=50")
      const history = await readApiJson<HistoryListResponse>(response)
      setHistoryItems(history.items)
      const nextRunId = selectedHistoryRunId || history.items[0]?.run_id || ""
      if (nextRunId) {
        setSelectedHistoryRunId(nextRunId)
        void loadHistoryDetail(nextRunId)
      } else {
        setHistoryDetail(null)
      }
    } catch (err) {
      if (isUnauthorizedError(err)) {
        markUnauthorized()
        setHistoryError("请输入 AgentArena access key。")
        return
      }
      const normalized = normalizeError(err)
      reportFrontendLog({
        level: "error",
        source: "app.history",
        message: normalized.message,
        stack: normalized.stack,
      })
      setHistoryError(normalized.message)
    } finally {
      setHistoryLoading(false)
    }
  }

  async function loadHistoryDetail(runId: string) {
    if (!runId) return
    setHistoryDetailLoading(true)
    setHistoryError("")
    try {
      const response = await apiFetch(`/history/${encodeURIComponent(runId)}`)
      setHistoryDetail(await readApiJson<HistoryDetailResponse>(response))
    } catch (err) {
      if (isUnauthorizedError(err)) {
        markUnauthorized()
        setHistoryError("请输入 AgentArena access key。")
        return
      }
      const normalized = normalizeError(err)
      reportFrontendLog({
        level: "error",
        source: "app.history_detail",
        message: normalized.message,
        stack: normalized.stack,
        payload: { run_id: runId },
      })
      setHistoryError(normalized.message)
    } finally {
      setHistoryDetailLoading(false)
    }
  }

  function selectHistoryRun(runId: string) {
    setSelectedHistoryRunId(runId)
    void loadHistoryDetail(runId)
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const trimmed = input.trim()
    if (!trimmed) {
      setError("请输入要对比的 prompt。")
      return
    }
    setError("")
    setLoading(true)
    setResult(null)
    setJudge(null)
    try {
      const response = await apiFetch("/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input: trimmed,
          system,
          temperature: parsedTemperature,
          max_tokens: parsedMaxTokens,
        }),
      })
      setResult(await readApiJson<CompareResponse>(response))
    } catch (err) {
      if (isUnauthorizedError(err)) {
        markUnauthorized()
        setError("请输入 AgentArena access key。")
        return
      }
      const normalized = normalizeError(err)
      reportFrontendLog({
        level: "error",
        source: "app.compare",
        message: normalized.message,
        stack: normalized.stack,
        payload: { input_length: input.length },
      })
      setError(normalized.message)
    } finally {
      setLoading(false)
    }
  }

  async function runJudge() {
    if (!result) return
    setJudgeLoading(true)
    setJudge(null)
    try {
      const response = await apiFetch("/judge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input,
          system,
          run_id: result.run_id,
          agent_a: result.agents.a,
          agent_b: result.agents.b,
        }),
      })
      setJudge(await readApiJson<JudgeResponse>(response))
    } catch (err) {
      if (isUnauthorizedError(err)) {
        markUnauthorized()
        setJudge({
          ok: false,
          winner: null,
          summary: "",
          answer_scores: {},
          process_scores: {},
          strengths: {},
          weaknesses: {},
          recommendations: {},
          latency_ms: null,
          error: "请输入 AgentArena access key。",
          raw: {},
        })
        return
      }
      const normalized = normalizeError(err)
      reportFrontendLog({
        level: "error",
        source: "app.judge",
        message: normalized.message,
        stack: normalized.stack,
      })
      setJudge({
        ok: false,
        winner: null,
        summary: "",
        answer_scores: {},
        process_scores: {},
        strengths: {},
        weaknesses: {},
        recommendations: {},
        latency_ms: null,
        error: normalized.message,
        raw: {},
      })
    } finally {
      setJudgeLoading(false)
    }
  }

  function clearAll() {
    setInput("")
    setSystem("")
    setResult(null)
    setJudge(null)
    setError("")
  }

  return (
    <main className="min-h-screen bg-muted/30">
      <div className="mx-auto flex w-full max-w-[1440px] flex-col gap-5 px-4 py-5 lg:px-6">
        <header className="flex flex-col gap-4 rounded-lg border bg-background px-5 py-4 shadow-sm lg:flex-row lg:items-center lg:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Swords className="size-5 text-primary" />
              <h1 className="text-xl font-semibold tracking-normal">Agent Arena</h1>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              同一输入，并发调用两个 Agent，展示最终答案、公开过程事件，并用 Judge 评估。
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              type="button"
              variant={view === "arena" ? "default" : "outline"}
              size="sm"
              onClick={() => setView("arena")}
            >
              <Swords className="size-4" />
              竞技场
            </Button>
            <Button
              type="button"
              variant={view === "batch" ? "default" : "outline"}
              size="sm"
              onClick={() => setView("batch")}
            >
              <FlaskConical className="size-4" />
              稳定性测试
            </Button>
            <Button
              type="button"
              variant={view === "history" ? "default" : "outline"}
              size="sm"
              onClick={() => setView("history")}
            >
              <History className="size-4" />
              历史记录
            </Button>
            <Badge variant="outline" className="gap-1">
              <Settings2 className="size-3" />
              Agent timeout {config?.timeout_seconds ?? "--"}s
            </Badge>
            <Badge variant="outline">Judge timeout {config?.judge_timeout_seconds ?? "--"}s</Badge>
            <Button type="button" variant="outline" size="sm" onClick={loadConfig}>
              <RefreshCcw className="size-4" />
              刷新配置
            </Button>
            <Button type="button" variant="outline" size="sm" onClick={openAccessKeyPrompt}>
              <KeyRound className="size-4" />
              Access key
            </Button>
          </div>
        </header>

        {needsAccessKey ? (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <KeyRound className="size-4 text-primary" />
                AgentArena 访问密钥
              </CardTitle>
              <CardDescription>
                后端设置了 ARENA_API_KEY，浏览器请求需要携带同一个 Bearer token。
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form className="flex flex-col gap-3 sm:flex-row sm:items-end" onSubmit={saveAccessKey}>
                <div className="min-w-0 flex-1 space-y-2">
                  <Label htmlFor="arenaAccessKey">Access key</Label>
                  <Input
                    id="arenaAccessKey"
                    type="password"
                    value={accessKeyInput}
                    onChange={(event) => setAccessKeyInput(event.target.value)}
                    placeholder="ARENA_API_KEY"
                    autoComplete="off"
                  />
                </div>
                <div className="flex gap-2">
                  <Button type="submit">
                    <KeyRound className="size-4" />
                    保存
                  </Button>
                  <Button type="button" variant="outline" onClick={resetAccessKey}>
                    <LogOut className="size-4" />
                    清除
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        ) : null}

        {configError && !needsAccessKey ? (
          <Alert variant="destructive">
            <AlertTriangle className="size-4" />
            <AlertTitle>配置读取失败</AlertTitle>
            <AlertDescription>{configError}</AlertDescription>
          </Alert>
        ) : null}

        {view === "batch" ? (
          <BatchPage config={config} />
        ) : view === "arena" ? (
        <section className="grid gap-5 xl:grid-cols-[420px_1fr]">
          <Card className="h-fit">
            <CardHeader>
              <CardTitle>输入</CardTitle>
              <CardDescription>
                请求由后端转发，浏览器不会接触 Agent 或 Judge API key。
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form className="space-y-4" onSubmit={submit}>
                <div className="space-y-2">
                  <Label htmlFor="prompt">User prompt</Label>
                  <Textarea
                    id="prompt"
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    placeholder="输入要同时发送给两个 Agent 的问题"
                    className="min-h-[180px] resize-y"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="system">System prompt</Label>
                  <Textarea
                    id="system"
                    value={system}
                    onChange={(event) => setSystem(event.target.value)}
                    placeholder="可选"
                    className="min-h-[96px] resize-y"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <Label htmlFor="temperature">Temperature</Label>
                    <Input
                      id="temperature"
                      value={temperature}
                      onChange={(event) => setTemperature(event.target.value)}
                      inputMode="decimal"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="maxTokens">Max tokens</Label>
                    <Input
                      id="maxTokens"
                      value={maxTokens}
                      onChange={(event) => setMaxTokens(event.target.value)}
                      inputMode="numeric"
                    />
                  </div>
                </div>
                {error ? (
                  <Alert variant="destructive">
                    <AlertTriangle className="size-4" />
                    <AlertTitle>提交失败</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                ) : null}
                <div className="flex flex-wrap gap-2">
                  <Button type="submit" disabled={loading || judgeLoading}>
                    {loading ? <Loader2 className="size-4 animate-spin" /> : <SendHorizontal className="size-4" />}
                    开始对比
                  </Button>
                  <Button type="button" variant="secondary" onClick={runJudge} disabled={!canJudge || judgeLoading}>
                    {judgeLoading ? <Loader2 className="size-4 animate-spin" /> : <Gavel className="size-4" />}
                    对比
                  </Button>
                  <Button type="button" variant="outline" onClick={clearAll} disabled={loading || judgeLoading}>
                    <Eraser className="size-4" />
                    清空
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>

          <div className="space-y-5">
            <div className="grid gap-5 lg:grid-cols-2">
              <AgentResultPanel title="Agent A" config={config?.agents.a} result={result?.agents.a} loading={loading} />
              <AgentResultPanel title="Agent B" config={config?.agents.b} result={result?.agents.b} loading={loading} />
            </div>

            <JudgePanel judge={judge} judgeLoading={judgeLoading} canJudge={canJudge} config={config} onJudge={runJudge} />

            <Card>
              <CardHeader className="pb-3">
                <CardTitle>调试信息</CardTitle>
                <CardDescription>查看实际请求 payload、Agent 返回和 Judge 返回。</CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="request">
                  <TabsList>
                    <TabsTrigger value="request">Request</TabsTrigger>
                    <TabsTrigger value="response">Response</TabsTrigger>
                    <TabsTrigger value="judge">Judge</TabsTrigger>
                    <TabsTrigger value="config">Config</TabsTrigger>
                  </TabsList>
                  <Separator className="my-3" />
                  <TabsContent value="request">
                    <CodeBlock
                      value={JSON.stringify(
                        { input, system, temperature: parsedTemperature, max_tokens: parsedMaxTokens },
                        null,
                        2,
                      )}
                    />
                  </TabsContent>
                  <TabsContent value="response">
                    <CodeBlock value={result ? JSON.stringify(result, null, 2) : "尚无对比结果"} />
                  </TabsContent>
                  <TabsContent value="judge">
                    <CodeBlock value={judge ? JSON.stringify(judge, null, 2) : "尚无 Judge 结果"} />
                  </TabsContent>
                  <TabsContent value="config">
                    <CodeBlock value={config ? JSON.stringify(config, null, 2) : "配置尚未加载"} />
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>
        </section>
        ) : (
          <HistoryPage
            items={historyItems}
            detail={historyDetail}
            selectedRunId={selectedHistoryRunId}
            loading={historyLoading}
            detailLoading={historyDetailLoading}
            error={historyError}
            config={config}
            onRefresh={loadHistory}
            onSelect={selectHistoryRun}
          />
        )}
      </div>
    </main>
  )
}

function CodeBlock({ value, compact = false }: { value: string; compact?: boolean }) {
  return (
    <ScrollArea className={cn(compact ? "h-[180px]" : "h-[240px]", "rounded-md border bg-slate-950 p-4")}>
      <pre className="whitespace-pre-wrap break-words text-xs leading-5 text-slate-100">{value}</pre>
    </ScrollArea>
  )
}

export default App
