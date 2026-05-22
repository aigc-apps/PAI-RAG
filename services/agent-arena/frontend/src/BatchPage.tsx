import { FormEvent, useMemo, useRef, useState } from "react"
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Download,
  Loader2,
  Play,
  Square,
  XCircle,
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import type { ConfigResponse } from "@/lib/types"
import { apiHeaders, apiUrl } from "@/lib/api"
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
  content_hash: string
  finish_reason: string | null
  error: string | null
  assertion_passed: boolean | null
  trace_summary: TraceSummary
}

type BatchCluster = {
  hash: string
  count: number
  sample: string
  indexes: number[]
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
  cluster_count: number
  clusters: BatchCluster[]
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

function StatusIcon({ item }: { item: BatchRunItem }) {
  if (!item.ok) return <XCircle className="size-4 text-destructive" />
  if (item.assertion_passed === false) {
    return <AlertTriangle className="size-4 text-amber-500" />
  }
  return <CheckCircle2 className="size-4 text-emerald-600" />
}

function SummaryCard({
  summary,
  hasAssertion,
}: {
  summary: BatchAgentSummary | undefined
  hasAssertion: boolean
}) {
  if (!summary) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">等待结果</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          尚未收到该 Agent 的完成事件。
        </CardContent>
      </Card>
    )
  }
  const successPct = summary.total ? summary.success / summary.total : 0
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between text-base">
          <span className="truncate">{summary.agent_name}</span>
          <Badge variant="outline" className="ml-2 shrink-0">
            {summary.agent_model || "—"}
          </Badge>
        </CardTitle>
        <CardDescription>{summary.total} 次执行</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="grid grid-cols-2 gap-3">
          <Metric label="成功率" value={`${(successPct * 100).toFixed(1)}% (${summary.success}/${summary.total})`} />
          <Metric
            label="断言通过率"
            value={
              hasAssertion
                ? `${formatPct(summary.assertion_rate)} (${summary.assertion_passed}/${summary.assertion_total})`
                : "—"
            }
          />
          <Metric label="延迟 min / p50" value={`${formatLatency(summary.latency_min_ms)} / ${formatLatency(summary.latency_p50_ms)}`} />
          <Metric label="延迟 p90 / max" value={`${formatLatency(summary.latency_p90_ms)} / ${formatLatency(summary.latency_max_ms)}`} />
          <Metric
            label="答案长度 min/avg/max"
            value={
              summary.content_len_avg === null
                ? "—"
                : `${summary.content_len_min ?? "—"} / ${summary.content_len_avg.toFixed(0)} / ${summary.content_len_max ?? "—"}`
            }
          />
          <Metric
            label="平均 tool 调用 / 失败"
            value={`${summary.tool_call_avg ?? "—"} / ${summary.failed_tool_total}`}
          />
          <Metric label="答案簇数" value={`${summary.cluster_count}`} />
          <Metric
            label="finish_reason"
            value={
              Object.keys(summary.finish_reasons).length
                ? Object.entries(summary.finish_reasons)
                    .map(([k, v]) => `${k}:${v}`)
                    .join(", ")
                : "—"
            }
          />
        </div>
        {summary.clusters.length ? (
          <div className="space-y-2">
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              答案指纹簇
            </div>
            <div className="space-y-2">
              {summary.clusters.map((cluster) => (
                <div key={cluster.hash} className="rounded-md border bg-muted/40 p-2 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[11px]">{cluster.hash}</span>
                    <Badge variant="secondary">{cluster.count} 次</Badge>
                  </div>
                  <p className="mt-1 line-clamp-3 whitespace-pre-wrap text-muted-foreground">
                    {cluster.sample}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border bg-background p-2">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 text-sm font-medium">{value}</div>
    </div>
  )
}

function ItemRow({
  item,
  expanded,
  onToggle,
}: {
  item: BatchRunItem
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <>
      <tr className="border-b last:border-b-0 hover:bg-muted/40">
        <td className="px-2 py-1.5">
          <button
            type="button"
            onClick={onToggle}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            aria-label={expanded ? "collapse" : "expand"}
          >
            {expanded ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
            <span className="font-mono">{item.index}</span>
          </button>
        </td>
        <td className="px-2 py-1.5 text-xs">
          <Badge variant="outline">{item.agent_key.toUpperCase()}</Badge>
        </td>
        <td className="px-2 py-1.5">
          <StatusIcon item={item} />
        </td>
        <td className="px-2 py-1.5 text-xs font-mono">{formatLatency(item.latency_ms)}</td>
        <td className="px-2 py-1.5 text-xs font-mono">{item.content_length}</td>
        <td className="px-2 py-1.5 text-xs">{item.finish_reason || "—"}</td>
        <td className="px-2 py-1.5 text-xs font-mono">{item.content_hash || "—"}</td>
        <td className="px-2 py-1.5 text-xs">
          {item.assertion_passed === null ? (
            <span className="text-muted-foreground">—</span>
          ) : item.assertion_passed ? (
            <CheckCircle2 className="size-4 text-emerald-600" />
          ) : (
            <XCircle className="size-4 text-destructive" />
          )}
        </td>
        <td className="max-w-[260px] truncate px-2 py-1.5 text-xs text-destructive">
          {previewError(item.error)}
        </td>
      </tr>
      {expanded ? (
        <tr className="border-b bg-muted/30">
          <td colSpan={9} className="px-3 py-2">
            <div className="space-y-2">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">Content</div>
              <ScrollArea className="h-[180px] rounded-md border bg-slate-950 p-3">
                <pre className="whitespace-pre-wrap break-words text-xs leading-5 text-slate-100">
                  {item.content || "(空)"}
                </pre>
              </ScrollArea>
              <div className="text-xs uppercase tracking-wide text-muted-foreground">Trace summary</div>
              <ScrollArea className="h-[120px] rounded-md border bg-slate-950 p-3">
                <pre className="whitespace-pre-wrap break-words text-xs leading-5 text-slate-100">
                  {JSON.stringify(item.trace_summary || {}, null, 2)}
                </pre>
              </ScrollArea>
              {item.error ? (
                <Alert variant="destructive">
                  <AlertTriangle className="size-4" />
                  <AlertTitle>错误</AlertTitle>
                  <AlertDescription className="whitespace-pre-wrap break-words">{item.error}</AlertDescription>
                </Alert>
              ) : null}
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

  return (
    <section className="grid gap-5 xl:grid-cols-[420px_1fr]">
      <Card className="h-fit">
        <CardHeader>
          <CardTitle>稳定性测试</CardTitle>
          <CardDescription>
            用同一个 query body 对单 Agent 跑 N 次，看成功率、延迟、答案是否稳定。
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={startBatch}>
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

            <div className="rounded-md border p-3">
              <label className="flex items-center gap-2 text-sm font-medium">
                <input
                  type="checkbox"
                  checked={assertionEnabled}
                  onChange={(e) => setAssertionEnabled(e.target.checked)}
                />
                启用 assertion
              </label>
              {assertionEnabled ? (
                <div className="mt-3 space-y-2">
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      size="sm"
                      variant={assertionKind === "substring" ? "default" : "outline"}
                      onClick={() => setAssertionKind("substring")}
                    >
                      substring
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant={assertionKind === "regex" ? "default" : "outline"}
                      onClick={() => setAssertionKind("regex")}
                    >
                      regex
                    </Button>
                    <label className="ml-auto flex items-center gap-1 text-xs">
                      <input
                        type="checkbox"
                        checked={assertionCaseSensitive}
                        onChange={(e) => setAssertionCaseSensitive(e.target.checked)}
                      />
                      区分大小写
                    </label>
                  </div>
                  <Input
                    placeholder={assertionKind === "regex" ? "JavaScript regex source" : "期望出现的子串"}
                    value={assertionValue}
                    onChange={(e) => setAssertionValue(e.target.value)}
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

            <div className="flex flex-wrap gap-2">
              {running ? (
                <Button type="button" variant="destructive" onClick={cancelBatch}>
                  <Square className="size-4" />
                  取消
                </Button>
              ) : (
                <Button type="submit">
                  <Play className="size-4" />
                  开始
                </Button>
              )}
              <Button type="button" variant="outline" disabled={!canExport} onClick={exportJson}>
                <Download className="size-4" />
                导出 JSON
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <div className="space-y-5">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center justify-between text-base">
              <span>执行进度</span>
              <div className="flex items-center gap-2 text-sm font-normal text-muted-foreground">
                {running ? <Loader2 className="size-4 animate-spin" /> : null}
                <span>
                  {items.length} / {total || "?"}
                </span>
                {batchId ? <Badge variant="outline">{batchId}</Badge> : null}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className={cn("h-full bg-primary transition-all", running ? "" : "")}
                style={{ width: `${Math.min(100, Math.round(progress * 100))}%` }}
              />
            </div>
            {completedAt ? (
              <p className="mt-2 text-xs text-muted-foreground">
                完成于 {new Date(completedAt).toLocaleString("zh-CN", { hour12: false })}
              </p>
            ) : null}
          </CardContent>
        </Card>

        <div className={cn("grid gap-5", targetKeys.length > 1 ? "lg:grid-cols-2" : "")}>
          {targetKeys.map((key) => (
            <SummaryCard key={key} summary={summaries[key]} hasAssertion={hasAssertion} />
          ))}
        </div>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">每次执行</CardTitle>
            <CardDescription>点击行号展开看完整 content 和 trace summary。</CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <Separator />
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-muted/40 text-xs uppercase tracking-wide text-muted-foreground">
                  <tr>
                    <th className="px-2 py-2">#</th>
                    <th className="px-2 py-2">agent</th>
                    <th className="px-2 py-2">状态</th>
                    <th className="px-2 py-2">延迟</th>
                    <th className="px-2 py-2">len</th>
                    <th className="px-2 py-2">finish</th>
                    <th className="px-2 py-2">hash</th>
                    <th className="px-2 py-2">assert</th>
                    <th className="px-2 py-2">error</th>
                  </tr>
                </thead>
                <tbody>
                  {items.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="px-2 py-6 text-center text-sm text-muted-foreground">
                        尚未开始或无结果。
                      </td>
                    </tr>
                  ) : (
                    items.map((item) => (
                      <ItemRow
                        key={rowKey(item)}
                        item={item}
                        expanded={expanded.has(rowKey(item))}
                        onToggle={() => toggleExpand(item)}
                      />
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  )
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
