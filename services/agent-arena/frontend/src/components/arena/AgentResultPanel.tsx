import { useState } from "react"
import {
  AlertTriangle,
  CheckCircle2,
  GitBranch,
  Loader2,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { AgentIdBadge } from "@/components/AgentIdBadge"
import { MetricGrid } from "@/components/MetricGrid"
import { StatusDot } from "@/components/StatusDot"
import { TraceModal } from "@/components/arena/TraceModal"
import { formatLatency, modelLabel } from "@/lib/formatters"
import type { AgentResult, PublicAgentConfig } from "@/lib/types"

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

export function AgentResultPanel({
  id,
  title,
  config,
  result,
  loading,
  input,
}: {
  id?: "a" | "b"
  title: string
  config?: PublicAgentConfig
  result?: AgentResult
  loading: boolean
  input?: string
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
  const traceAvailable = Boolean(result && (result.trace_events?.length || result.content || result.error))

  const [traceOpen, setTraceOpen] = useState(false)

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
        <div className="flex items-center justify-between border-b border-arena-border px-4 py-2">
          <div className="text-[12px] font-semibold text-arena-text-primary">答案</div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={!traceAvailable || loading}
            onClick={() => setTraceOpen(true)}
          >
            <GitBranch className="size-3.5" />
            查看 Trace
          </Button>
        </div>
        <div className="px-[18px] py-3.5">
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
        </div>
      </CardContent>

      <TraceModal
        open={traceOpen}
        onOpenChange={setTraceOpen}
        result={result || null}
        title={`Trace · ${displayName}`}
        subtitle={result?.model || ""}
        input={input}
      />
    </Card>
  )
}
