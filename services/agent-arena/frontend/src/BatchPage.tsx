import { FormEvent, useMemo, useRef, useState } from "react"
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Download,
  GitBranch,
  Play,
  Square,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { AgentIdBadge } from "@/components/AgentIdBadge"
import { StatCardRow, type StatItem } from "@/components/StatCard"
import { StatusDot } from "@/components/StatusDot"
import type {
  AgentResult,
  AgentTraceEvent,
  ConfigResponse,
  ConsistencyResponse,
} from "@/lib/types"
import { ConsistencyPanel } from "@/components/arena/ConsistencyPanel"
import { SaveToDatasetButton } from "@/components/arena/SaveToDatasetButton"
import { TraceModal } from "@/components/arena/TraceModal"
import { apiFetch, apiHeaders, apiUrl, readApiJson } from "@/lib/api"
import { normalizeError, reportFrontendLog } from "@/lib/frontendLogger"
import { cn } from "@/lib/utils"

type AgentKey = "a" | "b"
type Target = AgentKey | "both"
type Mode = "form" | "raw"
type AssertionKind = "substring" | "regex"

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

type BatchRunItem = {
  index: number
  agent_key: AgentKey
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
  trace_events?: AgentTraceEvent[]
}

type BatchAgentSummary = {
  agent_key: AgentKey
  agent_name: string
  agent_model: string
  total: number
  success: number
  success_rate: number
  assertion_total: number
  assertion_passed: number
  assertion_rate: number | null
  latency_min_ms: number | null
  latency_p50_ms: number | null
  latency_p90_ms: number | null
  latency_max_ms: number | null
  content_len_min: number | null
  content_len_avg: number | null
  content_len_max: number | null
  tool_call_avg: number | null
  failed_tool_total: number
  finish_reasons: Record<string, number>
}

type BatchStartedEvent = {
  type: "batch.started"
  batch_id: string
  created_at: string
  total: number
  target: Target
  mode: Mode
  iterations: number
  concurrency: number
  agents: Partial<Record<AgentKey, string>>
}

type BatchCompletedEvent = {
  type: "batch.completed"
  batch_id: string
  summaries: Partial<Record<AgentKey, BatchAgentSummary>>
  items_count: number
}

type BatchErrorEvent = { type: "batch.error"; error: string }
type RunCompletedEvent = { type: "run.completed"; item: BatchRunItem }

type BatchEvent =
  | BatchStartedEvent
  | RunCompletedEvent
  | BatchCompletedEvent
  | BatchErrorEvent

const DEFAULT_RAW_BODY = `{
  "model": "your-agent-model",
  "input": "请用三个要点说明长期记忆的设计取舍",
  "temperature": 0.2,
  "max_output_tokens": 1200
}`

const SAMPLE_FORM_INPUT =
  "请用三个要点说明：如果要给一个项目加入长期记忆能力，最重要的设计取舍是什么？"
const SAMPLE_FORM_SYSTEM = "你是一个严谨、直接的技术助手。"

function formatLatency(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—"
  if (value < 1000) return `${value} ms`
  return `${(value / 1000).toFixed(2)} s`
}

function formatPct(value: number | null | undefined, total = 1): string {
  if (value === null || value === undefined) return "—"
  return `${(value * (total === 1 ? 100 : 100 / total)).toFixed(1)}%`
}

function previewError(error: string | null): string {
  if (!error) return ""
  if (error.length <= 120) return error
  return `${error.slice(0, 120)}…`
}

async function consumeSse(
  response: Response,
  onEvent: (event: BatchEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  if (!response.body) return
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  try {
    while (true) {
      if (signal.aborted) {
        await reader.cancel().catch(() => undefined)
        return
      }
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let sepIndex
      while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, sepIndex)
        buffer = buffer.slice(sepIndex + 2)
        const lines = raw.split("\n")
        const data = lines
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).replace(/^ /, ""))
          .join("\n")
        if (!data) continue
        try {
          onEvent(JSON.parse(data))
        } catch {
          // ignore malformed SSE entry
        }
      }
    }
  } finally {
    reader.releaseLock?.()
  }
}

function rowStatusKind(item: BatchRunItem): "ok" | "warn" | "err" {
  if (!item.ok) return "err"
  if (item.assertion_passed === false) return "warn"
  return "ok"
}

function rowStatusLabel(item: BatchRunItem): string {
  if (!item.ok) return "failed"
  if (item.assertion_passed === false) return "degraded"
  return "ok"
}

function SummaryCard({
  agentKey,
  summary,
  hasAssertion,
  concurrency,
}: {
  agentKey: AgentKey
  summary: BatchAgentSummary | undefined
  hasAssertion: boolean
  concurrency: number
}) {
  if (!summary) {
    return (
      <Card>
        <CardHeader className="flex flex-row items-center gap-2.5">
          <AgentIdBadge id={agentKey} />
          <div className="min-w-0 flex-1">
            <CardTitle>等待 Agent {agentKey.toUpperCase()} 完成事件</CardTitle>
          </div>
        </CardHeader>
      </Card>
    )
  }
  const successPct = summary.total ? summary.success / summary.total : 0
  const finishText = Object.keys(summary.finish_reasons).length
    ? Object.entries(summary.finish_reasons)
        .map(([k, v]) => `${k} ×${v}`)
        .join(" · ")
    : "—"
  const stats: StatItem[] = [
    {
      label: "P50 延迟",
      value: formatLatency(summary.latency_p50_ms),
    },
    {
      label: "P90 延迟",
      value: formatLatency(summary.latency_p90_ms),
    },
    {
      label: "输出长度 avg",
      value: summary.content_len_avg === null ? "—" : Math.round(summary.content_len_avg),
      delta: {
        text:
          summary.content_len_min !== null && summary.content_len_max !== null
            ? `${summary.content_len_min} – ${summary.content_len_max}`
            : "",
        tone: "muted",
      },
    },
    {
      label: "平均 tool 调用",
      value: summary.tool_call_avg ?? "—",
      delta:
        summary.failed_tool_total > 0
          ? { text: `失败 ${summary.failed_tool_total}`, tone: "down" }
          : { text: "无失败", tone: "muted" },
    },
  ]
  return (
    <Card>
      <CardHeader className="flex flex-row items-center gap-2.5">
        <AgentIdBadge id={agentKey} />
        <div className="min-w-0 flex-1">
          <CardTitle className="truncate">{summary.agent_name}</CardTitle>
          <div className="mt-0.5 flex flex-wrap items-center gap-1.5 font-mono text-[11px] text-arena-text-tertiary">
            <span className="truncate">{summary.agent_model || "—"}</span>
            <span>·</span>
            <span>
              {summary.total} runs · concurrency {concurrency}
            </span>
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-1.5">
          <Badge variant={successPct >= 0.95 ? "success" : successPct >= 0.8 ? "warning" : "destructive"}>
            成功率 {(successPct * 100).toFixed(1)}%
          </Badge>
          {hasAssertion ? (
            <Badge
              variant={
                (summary.assertion_rate ?? 0) >= 0.9
                  ? "info"
                  : (summary.assertion_rate ?? 0) >= 0.6
                    ? "warning"
                    : "destructive"
              }
            >
              断言 {formatPct(summary.assertion_rate)}
            </Badge>
          ) : null}
        </div>
      </CardHeader>
      <StatCardRow items={stats} className="rounded-none border-x-0 border-b-0 shadow-none" />
      <CardContent className="bg-arena-bg-subtle p-[18px]">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[12.5px]">
          <KeyVal k="finish_reason" v={finishText} />
          <KeyVal
            k="断言通过"
            v={hasAssertion ? `${summary.assertion_passed}/${summary.assertion_total}` : "—"}
          />
          <KeyVal k="失败工具" v={summary.failed_tool_total} />
        </div>
      </CardContent>
    </Card>
  )
}

function KeyVal({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 font-mono">
      <span className="text-arena-text-tertiary">{k}</span>
      <span className="font-semibold text-arena-text-primary">{v}</span>
    </span>
  )
}

function ItemRow({
  item,
  batchId,
  batchInput,
  batchSystem,
  saveDisabled,
  expanded,
  onToggle,
  onOpenTrace,
}: {
  item: BatchRunItem
  batchId: string
  batchInput?: string
  batchSystem?: string
  saveDisabled: boolean
  expanded: boolean
  onToggle: () => void
  onOpenTrace: () => void
}) {
  const statusKind = rowStatusKind(item)
  const statusLabel = rowStatusLabel(item)
  return (
    <>
      <tr
        onClick={onToggle}
        className="cursor-pointer border-b border-arena-border bg-white last:border-b-0 hover:bg-arena-bg-hover"
      >
        <td className="px-1 py-1.5 text-arena-text-tertiary">
          <div className="flex items-center gap-0.5">
            {expanded ? (
              <ChevronDown className="size-3.5" />
            ) : (
              <ChevronRight className="size-3.5" />
            )}
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                onOpenTrace()
              }}
              className="inline-grid size-5 place-items-center rounded text-arena-text-tertiary transition-colors hover:bg-arena-accent-soft hover:text-arena-accent"
              title="查看 Trace"
            >
              <GitBranch className="size-3.5" />
            </button>
          </div>
        </td>
        <td className="px-2 py-1.5 font-mono text-[12px] text-arena-text-secondary">
          {String(item.index).padStart(3, "0")}
        </td>
        <td className="px-2 py-1.5">
          <div className="flex items-center gap-1.5 text-[12px] text-arena-text-primary">
            <AgentIdBadge id={item.agent_key} size="sm" />
            <span className="truncate">{item.agent_name}</span>
          </div>
        </td>
        <td className="px-2 py-1.5">
          <StatusDot kind={statusKind}>{statusLabel}</StatusDot>
        </td>
        <td className="px-2 py-1.5 text-right font-mono text-[12px] text-arena-text-primary">
          {formatLatency(item.latency_ms)}
        </td>
        <td className="px-2 py-1.5 text-right font-mono text-[12px] text-arena-text-primary">
          {item.content_length}
        </td>
        <td className="px-2 py-1.5 font-mono text-[12px] text-arena-text-secondary">
          {item.finish_reason || "—"}
        </td>
        <td className="px-2 py-1.5">
          {item.assertion_passed === null ? (
            <Badge variant="neutral">N/A</Badge>
          ) : item.assertion_passed ? (
            <Badge variant="success">PASS</Badge>
          ) : (
            <Badge variant="destructive">FAIL</Badge>
          )}
        </td>
        <td
          className="truncate px-2 py-1.5 font-mono text-[11px] text-arena-danger"
          title={item.error || undefined}
        >
          {previewError(item.error)}
        </td>
      </tr>
      {expanded ? (
        <tr className="border-b border-arena-border bg-arena-bg-subtle">
          <td colSpan={9} className="px-4 py-3">
            <div className="space-y-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-arena-text-tertiary">
                Content
              </div>
              <ScrollArea className="h-[180px] rounded-arena border border-arena-border bg-arena-bg-code p-3">
                <pre className="whitespace-pre-wrap break-words font-mono text-[12px] leading-5 text-slate-100">
                  {item.content || "(空)"}
                </pre>
              </ScrollArea>
              {item.error ? (
                <Alert variant="destructive">
                  <AlertTriangle className="size-4" />
                  <AlertTitle>错误</AlertTitle>
                  <AlertDescription className="whitespace-pre-wrap break-words">{item.error}</AlertDescription>
                </Alert>
              ) : null}
              <div className="flex items-center justify-between gap-2">
                <div className="text-[11px] text-arena-text-tertiary">
                  点击行首 <GitBranch className="inline size-3" /> 图标查看完整 Trace 可视化
                </div>
                <SaveToDatasetButton
                  source={{
                    kind: "batch-item",
                    batchId,
                    idx: item.index,
                    agentKey: item.agent_key,
                  }}
                  defaultQuery={batchInput}
                  defaultSystem={batchSystem}
                  defaultExpectedAnswer={item.content || ""}
                  label="此条入集"
                  disabled={saveDisabled}
                />
              </div>
            </div>
          </td>
        </tr>
      ) : null}
    </>
  )
}

export function BatchPage({ config }: { config: ConfigResponse | null }) {
  const [target, setTarget] = useState<Target>("a")
  const [mode, setMode] = useState<Mode>("form")
  const [iterations, setIterations] = useState("10")
  const [concurrency, setConcurrency] = useState("1")
  const [formInput, setFormInput] = useState(SAMPLE_FORM_INPUT)
  const [formSystem, setFormSystem] = useState(SAMPLE_FORM_SYSTEM)
  const [formTemperature, setFormTemperature] = useState("0.2")
  const [formMaxTokens, setFormMaxTokens] = useState("1200")
  const [rawBody, setRawBody] = useState(DEFAULT_RAW_BODY)
  const [assertionEnabled, setAssertionEnabled] = useState(false)
  const [assertionKind, setAssertionKind] = useState<AssertionKind>("substring")
  const [assertionValue, setAssertionValue] = useState("")
  const [assertionCaseSensitive, setAssertionCaseSensitive] = useState(false)

  const [running, setRunning] = useState(false)
  const [error, setError] = useState("")
  const [batchId, setBatchId] = useState("")
  const [total, setTotal] = useState(0)
  const [items, setItems] = useState<BatchRunItem[]>([])
  const [summaries, setSummaries] = useState<Partial<Record<AgentKey, BatchAgentSummary>>>({})
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [completedAt, setCompletedAt] = useState<string | null>(null)
  const [traceItem, setTraceItem] = useState<BatchRunItem | null>(null)
  const [consistencyResults, setConsistencyResults] = useState<ConsistencyResponse[]>([])

  const abortRef = useRef<AbortController | null>(null)

  const parsedIterations = useMemo(() => {
    const n = Number.parseInt(iterations, 10)
    if (!Number.isFinite(n) || n < 1) return 1
    return Math.min(200, n)
  }, [iterations])

  const parsedConcurrency = useMemo(() => {
    const n = Number.parseInt(concurrency, 10)
    if (!Number.isFinite(n) || n < 1) return 1
    return Math.min(8, n)
  }, [concurrency])

  const agentAName = config?.agents.a.name || "Agent A"
  const agentBName = config?.agents.b.name || "Agent B"

  const hasAssertion = assertionEnabled && assertionValue.trim().length > 0

  const targetKeys: AgentKey[] = target === "both" ? ["a", "b"] : [target]

  function rowKey(item: BatchRunItem) {
    return `${item.agent_key}-${item.index}`
  }

  function toggleExpand(item: BatchRunItem) {
    setExpanded((prev) => {
      const next = new Set(prev)
      const key = rowKey(item)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  function resetState() {
    setError("")
    setBatchId("")
    setTotal(0)
    setItems([])
    setSummaries({})
    setExpanded(new Set())
    setCompletedAt(null)
    setConsistencyResults([])
  }

  async function runConsistencyEval() {
    if (!batchId) return
    const response = await apiFetch(
      `/batch/${encodeURIComponent(batchId)}/consistency`,
      { method: "POST" },
    )
    const result = await readApiJson<ConsistencyResponse>(response)
    setConsistencyResults((prev) => [result, ...prev])
  }

  async function startBatch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (running) return

    let parsedRaw: Record<string, unknown> | null = null
    if (mode === "form") {
      if (!formInput.trim()) {
        setError("Form 模式下 user prompt 不能为空。")
        return
      }
    } else {
      try {
        const parsed = JSON.parse(rawBody)
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("raw body 必须是一个 JSON 对象")
        }
        parsedRaw = parsed as Record<string, unknown>
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        setError(`Raw JSON 无法解析：${msg}`)
        return
      }
    }
    if (hasAssertion && assertionKind === "regex") {
      try {
        // eslint-disable-next-line no-new
        new RegExp(assertionValue)
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        setError(`Regex 不合法：${msg}`)
        return
      }
    }

    const body: Record<string, unknown> = {
      target,
      mode,
      iterations: parsedIterations,
      concurrency: parsedConcurrency,
    }
    if (mode === "form") {
      const temperature = Number.parseFloat(formTemperature)
      const maxTokens = Number.parseInt(formMaxTokens, 10)
      body.form = {
        input: formInput.trim(),
        system: formSystem.trim(),
        temperature: Number.isFinite(temperature) ? Math.min(2, Math.max(0, temperature)) : 0.2,
        max_tokens: Number.isFinite(maxTokens) && maxTokens > 0 ? maxTokens : null,
      }
    } else {
      body.raw_body = parsedRaw
    }
    if (hasAssertion) {
      body.assertion = {
        type: assertionKind,
        value: assertionValue,
        case_sensitive: assertionCaseSensitive,
      }
    }

    resetState()
    setRunning(true)
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const response = await fetch(apiUrl("/batch"), {
        method: "POST",
        headers: apiHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(body),
        signal: controller.signal,
      })
      if (!response.ok) {
        const text = await response.text().catch(() => "")
        throw new Error(text || `HTTP ${response.status}`)
      }
      await consumeSse(
        response,
        (evt) => {
          if (evt.type === "batch.started") {
            setBatchId(evt.batch_id)
            setTotal(evt.total)
          } else if (evt.type === "run.completed") {
            setItems((prev) => [...prev, evt.item])
          } else if (evt.type === "batch.completed") {
            setSummaries(evt.summaries)
            setCompletedAt(new Date().toISOString())
          } else if (evt.type === "batch.error") {
            setError(evt.error)
          }
        },
        controller.signal,
      )
    } catch (err) {
      if (controller.signal.aborted) {
        setError("已取消，已完成的 run 仍保留在表格里。")
      } else {
        const normalized = normalizeError(err)
        reportFrontendLog({
          level: "error",
          source: "app.batch",
          message: normalized.message,
          stack: normalized.stack,
        })
        setError(normalized.message)
      }
    } finally {
      setRunning(false)
      abortRef.current = null
    }
  }

  function cancelBatch() {
    abortRef.current?.abort()
  }

  function exportJson() {
    const payload = {
      batch_id: batchId,
      generated_at: new Date().toISOString(),
      request: {
        target,
        mode,
        iterations: parsedIterations,
        concurrency: parsedConcurrency,
        form:
          mode === "form"
            ? {
                input: formInput,
                system: formSystem,
                temperature: Number.parseFloat(formTemperature),
                max_tokens: Number.parseInt(formMaxTokens, 10) || null,
              }
            : null,
        raw_body: mode === "raw" ? safeParseRaw(rawBody) : null,
        assertion: hasAssertion
          ? { type: assertionKind, value: assertionValue, case_sensitive: assertionCaseSensitive }
          : null,
      },
      summaries,
      items,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = `${batchId || "batch"}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const progress = total > 0 ? items.length / total : 0
  const canExport = items.length > 0 && !running
  const inFlight = running && total > 0 ? Math.max(0, Math.min(parsedConcurrency, total - items.length)) : 0

  return (
    <div className="grid gap-5 xl:grid-cols-[320px_minmax(0,1fr)]">
      <Card className="h-fit">
        <CardHeader>
          <CardTitle>运行配置</CardTitle>
        </CardHeader>
        <CardContent className="p-[18px]">
          <form className="space-y-3.5" onSubmit={startBatch}>
            <div className="space-y-2">
              <Label>目标 Agent</Label>
              <div className="grid grid-cols-3 gap-2">
                <TargetButton current={target} value="a" label={agentAName} onClick={setTarget} />
                <TargetButton current={target} value="b" label={agentBName} onClick={setTarget} />
                <TargetButton current={target} value="both" label="A + B" onClick={setTarget} />
              </div>
            </div>

            <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
              <TabsList className="w-full">
                <TabsTrigger value="form" className="flex-1">Form</TabsTrigger>
                <TabsTrigger value="raw" className="flex-1">Raw JSON</TabsTrigger>
              </TabsList>
              <TabsContent value="form" className="space-y-3">
                <div className="space-y-2">
                  <Label htmlFor="batchInput">User prompt</Label>
                  <Textarea
                    id="batchInput"
                    value={formInput}
                    onChange={(e) => setFormInput(e.target.value)}
                    className="min-h-[140px] resize-y"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="batchSystem">System prompt</Label>
                  <Textarea
                    id="batchSystem"
                    value={formSystem}
                    onChange={(e) => setFormSystem(e.target.value)}
                    className="min-h-[80px] resize-y"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <Label htmlFor="batchTemp">Temperature</Label>
                    <Input
                      id="batchTemp"
                      value={formTemperature}
                      onChange={(e) => setFormTemperature(e.target.value)}
                      inputMode="decimal"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="batchMax">Max tokens</Label>
                    <Input
                      id="batchMax"
                      value={formMaxTokens}
                      onChange={(e) => setFormMaxTokens(e.target.value)}
                      inputMode="numeric"
                    />
                  </div>
                </div>
              </TabsContent>
              <TabsContent value="raw" className="space-y-2">
                <Label htmlFor="batchRaw">Raw JSON body</Label>
                <Textarea
                  id="batchRaw"
                  value={rawBody}
                  onChange={(e) => setRawBody(e.target.value)}
                  className="min-h-[260px] resize-y font-mono text-xs"
                />
                <p className="text-xs text-muted-foreground">
                  原样 POST 到 Agent 的 <code>/v1/responses</code>，自动加 <code>stream: true</code>。
                </p>
              </TabsContent>
            </Tabs>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="batchIter">Iterations (≤200)</Label>
                <Input
                  id="batchIter"
                  value={iterations}
                  onChange={(e) => setIterations(e.target.value)}
                  inputMode="numeric"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="batchConc">Concurrency (≤8)</Label>
                <Input
                  id="batchConc"
                  value={concurrency}
                  onChange={(e) => setConcurrency(e.target.value)}
                  inputMode="numeric"
                />
              </div>
            </div>

            <div className="rounded-arena border border-arena-border bg-arena-bg-subtle p-3">
              <label className="flex items-center gap-2 text-[12.5px] font-semibold text-arena-text-primary">
                <input
                  type="checkbox"
                  checked={assertionEnabled}
                  onChange={(e) => setAssertionEnabled(e.target.checked)}
                  className="accent-arena-accent"
                />
                启用断言 (assertion)
              </label>
              {assertionEnabled ? (
                <div className="mt-3 space-y-2">
                  <div className="flex gap-1.5">
                    <Button
                      type="button"
                      size="sm"
                      variant={assertionKind === "substring" ? "default" : "outline"}
                      onClick={() => setAssertionKind("substring")}
                      className="flex-1"
                    >
                      substring
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant={assertionKind === "regex" ? "default" : "outline"}
                      onClick={() => setAssertionKind("regex")}
                      className="flex-1"
                    >
                      regex
                    </Button>
                  </div>
                  <label className="flex items-center gap-1.5 text-[11.5px] text-arena-text-secondary">
                    <input
                      type="checkbox"
                      checked={assertionCaseSensitive}
                      onChange={(e) => setAssertionCaseSensitive(e.target.checked)}
                      className="accent-arena-accent"
                    />
                    区分大小写
                  </label>
                  <Input
                    placeholder={assertionKind === "regex" ? "JavaScript regex source" : "期望出现的子串"}
                    value={assertionValue}
                    onChange={(e) => setAssertionValue(e.target.value)}
                    className="font-mono text-[12px]"
                  />
                </div>
              ) : null}
            </div>

            {error ? (
              <Alert variant="destructive">
                <AlertTriangle className="size-4" />
                <AlertTitle>提交失败</AlertTitle>
                <AlertDescription className="whitespace-pre-wrap">{error}</AlertDescription>
              </Alert>
            ) : null}

            <div className="flex flex-col gap-2 pt-1">
              {running ? (
                <Button type="button" variant="destructive" onClick={cancelBatch} className="w-full">
                  <Square className="size-4" />
                  取消运行
                </Button>
              ) : (
                <Button type="submit" className="w-full">
                  <Play className="size-4" />
                  开始 Batch
                </Button>
              )}
              <Button
                type="button"
                variant="outline"
                disabled={!canExport}
                onClick={exportJson}
                className="w-full"
              >
                <Download className="size-4" />
                导出 JSON
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <div className="space-y-5">
        <Card>
          <CardContent className="p-[18px]">
            <div className="mb-2.5 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <StatusDot
                  kind={running ? "run" : completedAt ? "ok" : items.length > 0 ? "warn" : "idle"}
                >
                  {running ? "运行中" : completedAt ? "已完成" : items.length > 0 ? "已停止" : "待启动"}
                </StatusDot>
                {batchId ? (
                  <span className="font-mono text-[11px] text-arena-text-tertiary">
                    batch_id: {batchId}
                  </span>
                ) : null}
              </div>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12.5px]">
                <KeyVal k="完成" v={`${items.length} / ${total || "?"}`} />
                {running ? <KeyVal k="in-flight" v={inFlight} /> : null}
                {completedAt ? (
                  <KeyVal
                    k="完成于"
                    v={new Date(completedAt).toLocaleString("zh-CN", { hour12: false })}
                  />
                ) : null}
              </div>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-arena-neutral-soft">
              <div
                className="h-full rounded-full bg-[linear-gradient(90deg,#FF5A1F_0%,#E84818_100%)] transition-all duration-200"
                style={{ width: `${Math.min(100, Math.round(progress * 100))}%` }}
              />
            </div>
          </CardContent>
        </Card>

        <div className={cn("grid gap-5", targetKeys.length > 1 ? "xl:grid-cols-2" : "")}>
          {targetKeys.map((key) => (
            <SummaryCard
              key={key}
              agentKey={key}
              summary={summaries[key]}
              hasAssertion={hasAssertion}
              concurrency={parsedConcurrency}
            />
          ))}
        </div>

        {batchId && !running && items.length > 0 ? (
          <ConsistencyPanel
            consistency={consistencyResults[0] ?? null}
            history={consistencyResults}
            onRun={runConsistencyEval}
          />
        ) : null}

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <div>
              <CardTitle>执行明细</CardTitle>
              <p className="mt-0.5 text-[11.5px] text-arena-text-tertiary">
                {items.length} 条记录 · 点击行可展开看完整 content 和 trace summary
              </p>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full table-fixed border-separate border-spacing-0 text-left">
                <colgroup>
                  <col className="w-[48px]" />
                  <col className="w-[56px]" />
                  <col className="w-[180px]" />
                  <col className="w-[96px]" />
                  <col className="w-[78px]" />
                  <col className="w-[64px]" />
                  <col className="w-[78px]" />
                  <col className="w-[74px]" />
                  <col />
                </colgroup>
                <thead className="bg-arena-bg-subtle text-[11px] font-semibold uppercase tracking-wider text-arena-text-tertiary">
                  <tr>
                    <th className="border-b border-arena-border px-2 py-2"></th>
                    <th className="border-b border-arena-border px-2 py-2">#</th>
                    <th className="border-b border-arena-border px-2 py-2">Agent</th>
                    <th className="border-b border-arena-border px-2 py-2">状态</th>
                    <th className="border-b border-arena-border px-2 py-2 text-right">延迟</th>
                    <th className="border-b border-arena-border px-2 py-2 text-right">长度</th>
                    <th className="border-b border-arena-border px-2 py-2">finish</th>
                    <th className="border-b border-arena-border px-2 py-2">断言</th>
                    <th className="border-b border-arena-border px-2 py-2">error</th>
                  </tr>
                </thead>
                <tbody>
                  {items.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="px-2 py-10 text-center text-[13px] text-arena-text-tertiary">
                        尚未开始或无结果。
                      </td>
                    </tr>
                  ) : (
                    items.map((item) => (
                      <ItemRow
                        key={rowKey(item)}
                        item={item}
                        batchId={batchId}
                        batchInput={mode === "form" ? formInput : undefined}
                        batchSystem={mode === "form" ? formSystem : undefined}
                        saveDisabled={running || !batchId}
                        expanded={expanded.has(rowKey(item))}
                        onToggle={() => toggleExpand(item)}
                        onOpenTrace={() => setTraceItem(item)}
                      />
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>

      <TraceModal
        open={traceItem !== null}
        onOpenChange={(open) => {
          if (!open) setTraceItem(null)
        }}
        result={traceItem ? batchItemToAgentResult(traceItem) : null}
        title={traceItem ? `Trace · #${String(traceItem.index).padStart(3, "0")} · ${traceItem.agent_name}` : "Trace"}
        subtitle={traceItem ? traceItem.agent_model : ""}
        input={mode === "form" ? formInput : undefined}
      />
    </div>
  )
}

function batchItemToAgentResult(item: BatchRunItem): AgentResult {
  return {
    ok: item.ok,
    name: item.agent_name,
    model: item.agent_model,
    content: item.content,
    latency_ms: item.latency_ms,
    error: item.error,
    raw_finish_reason: item.finish_reason,
    trace_supported: true,
    trace_events: item.trace_events || [],
    trace_summary: item.trace_summary,
  }
}

function TargetButton({
  current,
  value,
  label,
  onClick,
}: {
  current: Target
  value: Target
  label: string
  onClick: (v: Target) => void
}) {
  return (
    <Button
      type="button"
      variant={current === value ? "default" : "outline"}
      size="sm"
      className="truncate"
      onClick={() => onClick(value)}
    >
      {label}
    </Button>
  )
}

function safeParseRaw(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

export default BatchPage
