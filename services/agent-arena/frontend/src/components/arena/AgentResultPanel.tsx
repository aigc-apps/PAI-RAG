import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Loader2,
  Wrench,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AgentIdBadge } from "@/components/AgentIdBadge"
import { MetricGrid } from "@/components/MetricGrid"
import { StatusDot } from "@/components/StatusDot"
import { formatLatency, modelLabel } from "@/lib/formatters"
import { cn } from "@/lib/utils"
import type { AgentResult, AgentTraceEvent, PublicAgentConfig } from "@/lib/types"

export function AgentStatusBadge({ agent }: { agent?: PublicAgentConfig }) {
  if (!agent) return <Badge variant="neutral">加载中</Badge>
  if (!agent.configured) return <Badge variant="warning">未配置</Badge>
  return (
    <Badge variant={agent.has_api_key ? "success" : "neutral"}>
      {agent.has_api_key ? "已配置 key" : "无 key"}
    </Badge>
  )
}

export function ResultStatus({
  result,
  loading,
}: {
  result?: AgentResult
  loading: boolean
}) {
  if (loading) {
    return (
      <StatusDot kind="run">
        <span className="inline-flex items-center gap-1">
          <Loader2 className="size-3 animate-spin" />
          请求中
        </span>
      </StatusDot>
    )
  }
  if (!result) return <StatusDot kind="idle">待提交</StatusDot>
  return result.ok ? (
    <StatusDot kind="ok">
      <span className="inline-flex items-center gap-1">
        <CheckCircle2 className="size-3" />
        已完成
      </span>
    </StatusDot>
  ) : (
    <StatusDot kind="err">
      <span className="inline-flex items-center gap-1">
        <AlertTriangle className="size-3" />
        失败
      </span>
    </StatusDot>
  )
}

export function ResultBadge({ result, loading }: { result?: AgentResult; loading: boolean }) {
  return <ResultStatus result={result} loading={loading} />
}

export function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-arena border border-arena-border bg-arena-bg-subtle px-3 py-2">
      <div className="text-[11px] text-arena-text-tertiary">{label}</div>
      <div className="mt-1 truncate font-mono text-sm font-semibold text-arena-text-primary">
        {value}
      </div>
    </div>
  )
}

export function EmptyBox({ text }: { text: string }) {
  return (
    <div className="flex h-[260px] items-center justify-center rounded-arena border border-dashed border-arena-border bg-arena-bg-subtle text-sm text-arena-text-tertiary">
      {text}
    </div>
  )
}

function markerKindForEvent(event: AgentTraceEvent): {
  kind: "tool" | "think" | "msg" | "err"
  icon: React.ReactNode
} {
  const isError = Boolean(event.error) || event.event === "run.failed" || event.event.endsWith(".failed")
  if (isError) return { kind: "err", icon: <span className="text-[10px]">!</span> }
  if (event.event.startsWith("tool.")) return { kind: "tool", icon: <Wrench className="size-3" /> }
  if (event.event.startsWith("reasoning.")) return { kind: "think", icon: <Brain className="size-3" /> }
  return { kind: "msg", icon: <CheckCircle2 className="size-3" /> }
}

const MARKER_BG = {
  tool: "bg-arena-info",
  think: "bg-[#7C3AED]",
  msg: "bg-arena-success",
  err: "bg-arena-danger",
} as const

function TraceEventRow({ event, index }: { event: AgentTraceEvent; index: number }) {
  const { kind, icon } = markerKindForEvent(event)
  const time = typeof event.timestamp === "number" ? `+${event.timestamp.toFixed(2)}s` : `#${index + 1}`
  return (
    <div className="grid grid-cols-[24px_72px_1fr_auto] items-start gap-2.5 border-b border-dashed border-arena-border py-2 last:border-b-0">
      <div
        className={cn(
          "mt-0.5 grid size-[18px] place-items-center rounded-full text-white",
          MARKER_BG[kind],
        )}
      >
        {icon}
      </div>
      <div className="pt-0.5 font-mono text-[11px] text-arena-text-tertiary">{time}</div>
      <div className="min-w-0 text-[12.5px]">
        <div className="flex flex-wrap items-center gap-1.5 font-semibold text-arena-text-primary">
          <span>{event.event}</span>
          {event.tool ? (
            <code className="rounded bg-arena-neutral-soft px-1 py-px font-mono text-[11px] text-arena-text-primary">
              {event.tool}
            </code>
          ) : null}
        </div>
        {event.preview || event.text || event.delta ? (
          <pre className="mt-0.5 whitespace-pre-wrap break-words text-[12px] leading-snug text-arena-text-secondary">
            {event.preview || event.text || event.delta}
          </pre>
        ) : null}
      </div>
      <div className="pt-0.5">
        {event.duration !== null && event.duration !== undefined ? (
          <span className="rounded bg-arena-bg-subtle px-1.5 py-0.5 font-mono text-[11px] text-arena-text-tertiary">
            {event.duration}s
          </span>
        ) : null}
      </div>
    </div>
  )
}

export function TraceTimeline({ result, loading }: { result?: AgentResult; loading: boolean }) {
  if (loading) {
    return (
      <div className="space-y-3 px-1 py-2">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-10/12" />
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
    <ScrollArea className="h-[320px]">
      <div className="pr-1">
        {result.trace_events.map((event, index) => (
          <TraceEventRow event={event} index={index} key={`${event.event}-${index}`} />
        ))}
      </div>
    </ScrollArea>
  )
}

export function AgentResultPanel({
  id,
  title,
  config,
  result,
  loading,
}: {
  id?: "a" | "b"
  title: string
  config?: PublicAgentConfig
  result?: AgentResult
  loading: boolean
}) {
  const summary = result?.trace_summary || {}
  const displayName = config?.name || result?.name || title
  const displayModel = config ? modelLabel(config) : result?.model || "未加载配置"
  const latency = formatLatency(result?.latency_ms)
  const traceModeText =
    config?.trace_mode === "runs" || result?.trace_supported ? "runs trace" : "chat only"

  const eventCount = summary.event_count ?? result?.trace_events.length ?? 0
  const toolCount = summary.tool_call_count ?? 0
  const failedCount = summary.failed_tool_count ?? 0

  return (
    <Card className="flex flex-col">
      <CardHeader className="flex flex-row items-start gap-2.5">
        {id ? <AgentIdBadge id={id} className="mt-0.5" /> : null}
        <div className="min-w-0 flex-1">
          <CardTitle className="truncate">{displayName}</CardTitle>
          <div className="mt-0.5 flex flex-wrap items-center gap-1.5 font-mono text-[11px] text-arena-text-tertiary">
            <span className="truncate">{displayModel}</span>
            <span>·</span>
            <span>{traceModeText}</span>
            {config?.completion_path ? (
              <>
                <span>·</span>
                <span className="truncate">{config.completion_path}</span>
              </>
            ) : null}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <ResultStatus result={result} loading={loading} />
          {config ? <AgentStatusBadge agent={config} /> : null}
        </div>
      </CardHeader>

      <MetricGrid
        items={[
          { label: "延迟", value: latency === "未返回" ? "—" : latency },
          { label: "工具调用", value: toolCount },
          {
            label: "失败",
            value: failedCount,
            tone: failedCount > 0 ? "danger" : "muted",
          },
          { label: "事件", value: eventCount },
        ]}
      />

      <CardContent className="flex-1 p-0">
        <Tabs defaultValue="answer">
          <TabsList className="px-4">
            <TabsTrigger value="answer">答案</TabsTrigger>
            <TabsTrigger value="trace">
              过程
              <Badge variant="neutral" className="ml-1 px-1">
                {eventCount}
              </Badge>
            </TabsTrigger>
          </TabsList>
          <TabsContent value="answer" className="mt-0 px-[18px] py-3.5">
            {loading ? (
              <div className="space-y-3">
                <Skeleton className="h-4 w-11/12" />
                <Skeleton className="h-4 w-10/12" />
                <Skeleton className="h-4 w-9/12" />
                <Skeleton className="h-32 w-full" />
              </div>
            ) : result?.error ? (
              <Alert variant="destructive">
                <AlertTriangle className="size-4" />
                <AlertTitle>请求失败</AlertTitle>
                <AlertDescription>
                  <pre className="mt-1 max-h-[260px] whitespace-pre-wrap break-words font-mono text-[12px]">
                    {result.error}
                  </pre>
                </AlertDescription>
              </Alert>
            ) : result?.content ? (
              <ScrollArea className="h-[280px] pr-2">
                <pre className="whitespace-pre-wrap break-words text-[13px] leading-relaxed text-arena-text-primary">
                  {result.content}
                </pre>
              </ScrollArea>
            ) : (
              <EmptyBox text="等待输入并提交对比" />
            )}
          </TabsContent>
          <TabsContent value="trace" className="mt-0 px-[18px] py-3.5">
            <TraceTimeline result={result} loading={loading} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
