import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Loader2,
  Server,
  Wrench,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { formatLatency, modelLabel } from "@/lib/formatters"
import type { AgentResult, AgentTraceEvent, PublicAgentConfig } from "@/lib/types"

export function AgentStatusBadge({ agent }: { agent?: PublicAgentConfig }) {
  if (!agent) return <Badge variant="secondary">加载中</Badge>
  if (!agent.configured) return <Badge variant="warning">未配置</Badge>
  return (
    <Badge variant={agent.has_api_key ? "success" : "secondary"}>
      {agent.has_api_key ? "已配置 key" : "无 key"}
    </Badge>
  )
}

export function ResultBadge({ result, loading }: { result?: AgentResult; loading: boolean }) {
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

export function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border bg-muted/30 px-3 py-2">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="mt-1 truncate text-sm font-semibold">{value}</div>
    </div>
  )
}

export function EmptyBox({ text }: { text: string }) {
  return (
    <div className="flex h-[300px] items-center justify-center rounded-md border border-dashed bg-muted/20 text-sm text-muted-foreground">
      {text}
    </div>
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

export function TraceTimeline({ result, loading }: { result?: AgentResult; loading: boolean }) {
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

export function AgentResultPanel({
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
