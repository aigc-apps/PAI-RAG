import { useEffect, useMemo, useState } from "react"
import {
  AlertTriangle,
  Brain,
  Cpu,
  GitBranch,
  MessageSquare,
  Sparkles,
  Wrench,
} from "lucide-react"

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { cn } from "@/lib/utils"
import type { AgentResult, AgentTraceEvent } from "@/lib/types"

type SpanKind = "root" | "llm" | "tool" | "think" | "msg" | "err" | "lifecycle"

type SpanNode = {
  id: string
  kind: SpanKind
  name: string
  durationSec: number | null
  timestampSec: number | null
  event?: AgentTraceEvent
  depth: number
}

const KIND_META: Record<
  SpanKind,
  { label: string; icon: React.ReactNode; bar: string; chip: string }
> = {
  root: {
    label: "RUN",
    icon: <GitBranch className="size-3.5" />,
    bar: "bg-arena-accent",
    chip: "bg-arena-accent-soft text-arena-accent border-arena-accent/30",
  },
  llm: {
    label: "LLM",
    icon: <Cpu className="size-3.5" />,
    bar: "bg-arena-info",
    chip: "bg-arena-info-soft text-arena-info border-arena-info/30",
  },
  tool: {
    label: "TOOL",
    icon: <Wrench className="size-3.5" />,
    bar: "bg-amber-500",
    chip: "bg-amber-50 text-amber-700 border-amber-200",
  },
  think: {
    label: "THINK",
    icon: <Brain className="size-3.5" />,
    bar: "bg-violet-500",
    chip: "bg-violet-50 text-violet-700 border-violet-200",
  },
  msg: {
    label: "MSG",
    icon: <MessageSquare className="size-3.5" />,
    bar: "bg-emerald-500",
    chip: "bg-emerald-50 text-emerald-700 border-emerald-200",
  },
  err: {
    label: "ERR",
    icon: <AlertTriangle className="size-3.5" />,
    bar: "bg-arena-danger",
    chip: "bg-arena-danger-soft text-arena-danger border-arena-danger/30",
  },
  lifecycle: {
    label: "EVT",
    icon: <Sparkles className="size-3.5" />,
    bar: "bg-arena-text-tertiary",
    chip: "bg-arena-neutral-soft text-arena-text-tertiary border-arena-border",
  },
}

function classifyEvent(event: AgentTraceEvent): SpanKind {
  if (event.error || event.event.endsWith(".failed")) return "err"
  if (event.event.startsWith("tool.")) return "tool"
  if (event.event.startsWith("reasoning.")) return "think"
  if (event.event.startsWith("response.")) return "lifecycle"
  if (event.event.startsWith("run.")) return "lifecycle"
  if (event.event.includes("message") || event.event.includes("output_text")) return "msg"
  if (event.event.includes("completion") || event.event.includes("chat")) return "llm"
  return "msg"
}

function formatSeconds(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—"
  if (seconds >= 1) return `${seconds.toFixed(2)}s`
  return `${Math.round(seconds * 1000)}ms`
}

function formatLatencyMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—"
  return formatSeconds(ms / 1000)
}

function eventDisplayName(event: AgentTraceEvent): string {
  if (event.tool) return `${event.event} · ${event.tool}`
  return event.event
}

function eventBody(event: AgentTraceEvent): string {
  return event.text || event.delta || event.preview || ""
}

function deltaStem(eventName: string): string | null {
  if (eventName.endsWith(".delta")) return eventName.slice(0, -".delta".length)
  if (eventName.endsWith(".done")) return eventName.slice(0, -".done".length)
  return null
}

// Lifecycle / structural markers that carry no content — pure start/end
// scaffolding emitted by the Responses API. Filtered before span construction.
const NOISE_EVENTS = new Set([
  "response.created",
  "response.in_progress",
  "response.completed",
  "response.reasoning_step.started",
  "response.reasoning_step.completed",
  "response.output_item.added",
  "response.output_item.done",
  "response.content_part.added",
  "response.content_part.done",
])

function isNoiseEvent(ev: AgentTraceEvent): boolean {
  if (ev.error) return false
  return NOISE_EVENTS.has(ev.event)
}

// Merge consecutive *.delta events (optionally followed by *.done) into a
// single composite event whose `text` is the concatenated stream. Avoids the
// 100-row reasoning_text.delta wall and shows the full reasoning/answer in one
// expandable span.
function collapseDeltaRuns(events: AgentTraceEvent[]): AgentTraceEvent[] {
  const out: AgentTraceEvent[] = []
  let i = 0
  while (i < events.length) {
    const ev = events[i]
    const stem = deltaStem(ev.event)
    if (stem === null) {
      out.push(ev)
      i += 1
      continue
    }
    // collect contiguous events sharing this stem (any .delta or .done variant)
    const startTs = typeof ev.timestamp === "number" ? ev.timestamp : null
    let endTs = startTs
    let accText = ""
    let lastDone: AgentTraceEvent | null = null
    let lastErr: AgentTraceEvent["error"] = null
    let lastTool: string | null = null
    let j = i
    while (j < events.length && deltaStem(events[j].event) === stem) {
      const cur = events[j]
      const piece = cur.delta || cur.text || ""
      if (piece) accText += piece
      if (typeof cur.timestamp === "number") endTs = cur.timestamp
      if (cur.event.endsWith(".done")) {
        lastDone = cur
        const doneText = cur.text || ""
        if (doneText) accText = doneText
      }
      if (cur.error) lastErr = cur.error
      if (cur.tool) lastTool = cur.tool
      j += 1
    }
    const count = j - i
    const baseName = lastDone ? `${stem} (×${count})` : `${stem} (×${count})`
    out.push({
      event: baseName,
      timestamp: endTs,
      tool: lastTool,
      preview: accText ? accText.slice(0, 200) : null,
      text: accText || null,
      delta: null,
      duration:
        startTs !== null && endTs !== null && endTs >= startTs ? endTs - startTs : null,
      error: lastErr,
    })
    i = j
  }
  return out
}

function buildSpans(result: AgentResult): SpanNode[] {
  const rawEvents = (result.trace_events || []).filter((ev) => !isNoiseEvent(ev))
  const events = collapseDeltaRuns(rawEvents).filter((ev) => {
    // Drop collapsed delta runs whose accumulated text is whitespace-only —
    // these are usually the "\n" separators the agent emits between tool
    // calls and add no signal to the trace view.
    if (!ev.event.startsWith("response.") || ev.error) return true
    const body = (ev.text || ev.delta || "").trim()
    return body.length > 0
  })
  let maxTs = 0
  for (const ev of events) {
    const t = typeof ev.timestamp === "number" ? ev.timestamp : 0
    if (t > maxTs) maxTs = t
  }
  const totalSec = maxTs > 0 ? maxTs : (result.latency_ms ? result.latency_ms / 1000 : 0)

  const root: SpanNode = {
    id: "root",
    kind: "root",
    name: result.name || "agent_run",
    durationSec: totalSec || null,
    timestampSec: 0,
    depth: 0,
  }
  const children: SpanNode[] = events.map((ev, idx) => ({
    id: `ev-${idx}`,
    kind: classifyEvent(ev),
    name: eventDisplayName(ev),
    durationSec: typeof ev.duration === "number" ? ev.duration : null,
    timestampSec: typeof ev.timestamp === "number" ? ev.timestamp : null,
    event: ev,
    depth: 1,
  }))
  return [root, ...children]
}

function MetaPill({
  label,
  value,
  tone = "neutral",
}: {
  label: string
  value: React.ReactNode
  tone?: "neutral" | "success" | "danger" | "accent"
}) {
  const toneClasses: Record<string, string> = {
    neutral: "border-arena-border text-arena-text-primary bg-arena-bg-card",
    success: "border-arena-success/30 text-arena-success bg-arena-success-soft",
    danger: "border-arena-danger/30 text-arena-danger bg-arena-danger-soft",
    accent: "border-arena-accent/30 text-arena-accent bg-arena-accent-soft",
  }
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-arena border px-2 py-0.5 font-mono text-[11px]",
        toneClasses[tone],
      )}
    >
      <span className="text-arena-text-tertiary">{label}</span>
      <span className="font-semibold">{value}</span>
    </span>
  )
}

function SpanRow({
  span,
  totalSec,
  selected,
  onClick,
}: {
  span: SpanNode
  totalSec: number
  selected: boolean
  onClick: () => void
}) {
  const meta = KIND_META[span.kind]
  const ts = span.timestampSec ?? 0
  const dur = span.durationSec ?? 0
  let barLeft = 0
  let barWidth = 0
  if (totalSec > 0) {
    const start = dur > 0 ? Math.max(0, ts - dur) : ts
    barLeft = Math.min(100, (start / totalSec) * 100)
    barWidth = dur > 0 ? Math.max(0.6, (dur / totalSec) * 100) : 0.6
  }
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "group flex w-full flex-col gap-1 rounded-arena border px-2 py-1.5 text-left transition-colors",
        selected
          ? "border-arena-accent/40 bg-arena-accent-soft/50"
          : "border-transparent hover:bg-arena-bg-subtle",
      )}
      style={{ marginLeft: span.depth * 12 }}
    >
      <div className="flex items-center gap-1.5">
        <span
          className={cn(
            "inline-grid size-4 place-items-center rounded text-white",
            meta.bar,
          )}
        >
          {meta.icon}
        </span>
        <span className="min-w-0 flex-1 truncate text-[12.5px] font-semibold text-arena-text-primary">
          {span.name}
        </span>
        <span className="font-mono text-[10.5px] text-arena-text-tertiary">
          {formatSeconds(span.durationSec)}
        </span>
      </div>
      <div className="relative h-1.5 overflow-hidden rounded bg-arena-bg-subtle">
        {totalSec > 0 ? (
          <span
            className={cn("absolute top-0 h-full rounded", meta.bar)}
            style={{ left: `${barLeft}%`, width: `${barWidth}%` }}
          />
        ) : null}
      </div>
    </button>
  )
}

function Section({
  title,
  tone = "neutral",
  children,
}: {
  title: string
  tone?: "neutral" | "input" | "output" | "error"
  children: React.ReactNode
}) {
  const toneClasses: Record<string, string> = {
    neutral: "bg-arena-bg-subtle border-arena-border",
    input: "bg-arena-bg-subtle border-arena-border",
    output: "bg-arena-success-soft/40 border-arena-success/20",
    error: "bg-arena-danger-soft border-arena-danger/30",
  }
  return (
    <div className="space-y-1.5">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-arena-text-tertiary">
        {title}
      </div>
      <div
        className={cn(
          "rounded-arena border px-3 py-2 font-mono text-[12px] leading-relaxed text-arena-text-primary",
          toneClasses[tone],
        )}
      >
        {children}
      </div>
    </div>
  )
}

function SpanDetail({
  span,
  result,
  input,
}: {
  span: SpanNode
  result: AgentResult
  input?: string
}) {
  const meta = KIND_META[span.kind]
  const isRoot = span.kind === "root"
  const body = span.event ? eventBody(span.event) : ""
  const errMsg = span.event && typeof span.event.error === "string" ? span.event.error : null

  const rawJson = isRoot
    ? JSON.stringify(
        {
          name: result.name,
          model: result.model,
          ok: result.ok,
          latency_ms: result.latency_ms,
          finish_reason: result.raw_finish_reason,
          trace_summary: result.trace_summary,
        },
        null,
        2,
      )
    : JSON.stringify(span.event, null, 2)

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-arena-border bg-arena-bg-card px-4 py-2.5">
        <span
          className={cn(
            "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wide",
            meta.chip,
          )}
        >
          {meta.icon}
          {meta.label}
        </span>
        <span className="min-w-0 flex-1 truncate text-[13px] font-semibold text-arena-text-primary">
          {span.name}
        </span>
        <span className="font-mono text-[11px] text-arena-text-tertiary">
          {span.timestampSec !== null ? `+${formatSeconds(span.timestampSec)}` : ""}
          {span.durationSec !== null ? (
            <span className="ml-2 text-arena-text-primary">
              {formatSeconds(span.durationSec)}
            </span>
          ) : null}
        </span>
      </div>
      <Tabs defaultValue="preview" className="flex min-h-0 flex-1 flex-col">
        <TabsList className="px-4">
          <TabsTrigger value="preview">Preview</TabsTrigger>
          <TabsTrigger value="raw">Raw JSON</TabsTrigger>
        </TabsList>
        <TabsContent value="preview" className="mt-0 min-h-0 flex-1">
          <ScrollArea className="h-full">
            <div className="space-y-3 px-4 py-3">
              {isRoot ? (
                <>
                  {input ? (
                    <Section title="Input" tone="input">
                      <pre className="whitespace-pre-wrap break-words">{input}</pre>
                    </Section>
                  ) : null}
                  {result.content ? (
                    <Section title="Output" tone="output">
                      <pre className="whitespace-pre-wrap break-words">{result.content}</pre>
                    </Section>
                  ) : (
                    <Section title="Output">
                      <span className="text-arena-text-tertiary">(空)</span>
                    </Section>
                  )}
                  {result.error ? (
                    <Section title="Error" tone="error">
                      <pre className="whitespace-pre-wrap break-words text-arena-danger">
                        {result.error}
                      </pre>
                    </Section>
                  ) : null}
                  <Section title="Metadata">
                    <pre className="whitespace-pre-wrap break-words">
                      {JSON.stringify(
                        {
                          model: result.model,
                          latency_ms: result.latency_ms,
                          finish_reason: result.raw_finish_reason,
                          trace_summary: result.trace_summary,
                        },
                        null,
                        2,
                      )}
                    </pre>
                  </Section>
                </>
              ) : (
                <>
                  {span.event?.tool ? (
                    <Section title="Tool">
                      <code className="font-mono">{span.event.tool}</code>
                    </Section>
                  ) : null}
                  {body ? (
                    <Section
                      title={span.kind === "tool" ? "Output" : "Content"}
                      tone={span.kind === "msg" ? "output" : "neutral"}
                    >
                      <pre className="whitespace-pre-wrap break-words">{body}</pre>
                    </Section>
                  ) : (
                    <Section title="Content">
                      <span className="text-arena-text-tertiary">(无内容)</span>
                    </Section>
                  )}
                  {errMsg ? (
                    <Section title="Error" tone="error">
                      <pre className="whitespace-pre-wrap break-words text-arena-danger">
                        {errMsg}
                      </pre>
                    </Section>
                  ) : null}
                </>
              )}
            </div>
          </ScrollArea>
        </TabsContent>
        <TabsContent value="raw" className="mt-0 min-h-0 flex-1">
          <ScrollArea className="h-full">
            <pre className="whitespace-pre-wrap break-words bg-arena-bg-code px-4 py-3 font-mono text-[12px] leading-relaxed text-slate-100">
              {rawJson}
            </pre>
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export function TraceModal({
  open,
  onOpenChange,
  result,
  title,
  subtitle,
  input,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  result: AgentResult | null
  title: string
  subtitle?: string
  input?: string
}) {
  const spans = useMemo<SpanNode[]>(() => {
    if (!result) return []
    return buildSpans(result)
  }, [result])
  const totalSec = spans[0]?.durationSec || 0
  const [selectedId, setSelectedId] = useState<string>("root")
  const selected = spans.find((s) => s.id === selectedId) || spans[0]

  // reset selection when result changes
  useEffect(() => {
    setSelectedId("root")
  }, [result?.name, result?.content, result?.latency_ms])

  const summary = result?.trace_summary || {}
  const eventCount = summary.event_count ?? result?.trace_events.length ?? 0
  const toolCount = summary.tool_call_count ?? 0
  const failedCount = summary.failed_tool_count ?? 0

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="p-0">
        {result ? (
          <div className="flex flex-col">
            <div className="border-b border-arena-border bg-arena-bg-subtle px-4 py-3">
              <div className="flex items-start gap-2">
                <div className="min-w-0 flex-1">
                  <DialogTitle className="truncate">{title}</DialogTitle>
                  {subtitle ? (
                    <DialogDescription className="mt-0.5 truncate font-mono">
                      {subtitle}
                    </DialogDescription>
                  ) : null}
                </div>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-1.5">
                <MetaPill
                  label="状态"
                  value={result.ok ? "OK" : "FAILED"}
                  tone={result.ok ? "success" : "danger"}
                />
                <MetaPill label="Latency" value={formatLatencyMs(result.latency_ms)} />
                <MetaPill label="Events" value={eventCount} />
                <MetaPill label="Tools" value={toolCount} />
                {failedCount > 0 ? (
                  <MetaPill label="Failed" value={failedCount} tone="danger" />
                ) : null}
                <MetaPill label="Model" value={result.model || "—"} />
                {result.raw_finish_reason ? (
                  <MetaPill label="Finish" value={result.raw_finish_reason} />
                ) : null}
              </div>
            </div>
            <div className="grid min-h-[560px] grid-cols-[300px_1fr] max-h-[calc(92vh-130px)]">
              <div className="flex min-h-0 flex-col border-r border-arena-border bg-arena-bg-page">
                <div className="border-b border-arena-border px-3 py-1.5">
                  <div className="flex items-center justify-between font-mono text-[11px] text-arena-text-tertiary">
                    <span>Spans</span>
                    <span>{spans.length}</span>
                  </div>
                </div>
                <ScrollArea className="min-h-0 flex-1">
                  <div className="space-y-1 p-2">
                    {spans.map((span) => (
                      <SpanRow
                        key={span.id}
                        span={span}
                        totalSec={totalSec}
                        selected={span.id === selectedId}
                        onClick={() => setSelectedId(span.id)}
                      />
                    ))}
                    {spans.length === 1 ? (
                      <div className="px-2 py-3 text-center text-[11.5px] text-arena-text-tertiary">
                        无过程事件 — 仅展示根 span
                      </div>
                    ) : null}
                  </div>
                </ScrollArea>
              </div>
              <div className="min-h-0">
                {selected ? (
                  <SpanDetail span={selected} result={result} input={input} />
                ) : null}
              </div>
            </div>
          </div>
        ) : (
          <div className="px-6 py-10 text-center text-sm text-arena-text-tertiary">
            无可用 trace
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
