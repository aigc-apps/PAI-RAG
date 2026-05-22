import { AlertTriangle, Loader2, RefreshCcw } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AgentResultPanel, EmptyBox } from "@/components/arena/AgentResultPanel"
import { CodeBlock } from "@/components/arena/CodeBlock"
import { JudgePanel } from "@/components/arena/JudgePanel"
import { StatusDot } from "@/components/StatusDot"
import {
  formatDateTime,
  formatLatency,
  historyAgentName,
  historyAgentOk,
} from "@/lib/formatters"
import { cn } from "@/lib/utils"
import type {
  ConfigResponse,
  HistoryDetailResponse,
  HistorySummary,
} from "@/lib/types"

function relativeTime(value: string): string {
  if (!value) return ""
  const ts = new Date(value).getTime()
  if (Number.isNaN(ts)) return value
  const diff = Date.now() - ts
  if (diff < 60_000) return `${Math.max(1, Math.floor(diff / 1000))}s ago`
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`
  return `${Math.floor(diff / 86_400_000)}d ago`
}

function agentLatencyMs(summary: Record<string, unknown>): number | null {
  const value = summary.latency_ms
  if (typeof value === "number" && Number.isFinite(value)) return value
  return null
}

export function HistoryPage({
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
    <div className="grid gap-5 xl:grid-cols-[340px_minmax(0,1fr)]">
      <Card className="h-fit">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>对比历史</CardTitle>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onRefresh}
            disabled={loading}
          >
            {loading ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <RefreshCcw className="size-3.5" />
            )}
            刷新
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          {error ? (
            <div className="px-[18px] pt-3">
              <Alert variant="destructive">
                <AlertTriangle className="size-4" />
                <AlertTitle>历史读取失败</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            </div>
          ) : null}
          {loading ? (
            <div className="space-y-3 px-[18px] py-3">
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
            </div>
          ) : items.length ? (
            <ScrollArea className="h-[calc(100vh-theme(spacing.topbar)-200px)]">
              <div className="divide-y divide-arena-border">
                {items.map((item) => (
                  <HistoryListItem
                    item={item}
                    selected={item.run_id === selectedRunId}
                    onClick={() => onSelect(item.run_id)}
                    key={item.run_id}
                  />
                ))}
              </div>
            </ScrollArea>
          ) : (
            <div className="px-[18px] py-3">
              <EmptyBox text="还没有历史记录" />
            </div>
          )}
        </CardContent>
      </Card>

      <div className="space-y-5">
        {detailLoading ? (
          <div className="space-y-5">
            <Skeleton className="h-32 w-full" />
            <div className="grid gap-5 xl:grid-cols-2">
              <Skeleton className="h-[520px] w-full" />
              <Skeleton className="h-[520px] w-full" />
            </div>
          </div>
        ) : detail ? (
          <HistoryDetail detail={detail} config={config} />
        ) : (
          <Card>
            <CardContent className="p-[18px]">
              <EmptyBox text="从左侧选择一条历史记录查看详情" />
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

function WinnerBadge({
  winner,
}: {
  winner: "a" | "b" | "tie" | string | null | undefined
}) {
  if (winner === "tie") {
    return (
      <span className="rounded-arena-sm bg-arena-neutral px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider text-white">
        TIE
      </span>
    )
  }
  if (winner === "a") {
    return (
      <span className="rounded-arena-sm bg-arena-accent px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider text-white">
        WIN · A
      </span>
    )
  }
  if (winner === "b") {
    return (
      <span className="rounded-arena-sm bg-arena-success px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider text-white">
        WIN · B
      </span>
    )
  }
  return <Badge variant="neutral">未评估</Badge>
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
  const aLatency = agentLatencyMs(item.agent_a)
  const bLatency = agentLatencyMs(item.agent_b)
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "block w-full px-[18px] py-3 text-left transition-colors hover:bg-arena-bg-hover",
        selected && "bg-arena-accent-soft hover:bg-arena-accent-soft",
      )}
    >
      <div className="flex flex-wrap items-center gap-2">
        <WinnerBadge winner={latestJudge?.winner} />
        {item.judge_count ? (
          <Badge variant="neutral">Judge ×{item.judge_count}</Badge>
        ) : null}
        <span className="ml-auto font-mono text-[10.5px] text-arena-text-tertiary">
          {relativeTime(item.created_at)}
        </span>
      </div>
      <div className="mt-2 line-clamp-2 text-[13px] leading-snug text-arena-text-primary">
        {item.input}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-3">
        <StatusDot kind={agentAOk ? "ok" : "err"}>
          A {agentAOk ? formatLatency(aLatency) : "failed"}
        </StatusDot>
        <StatusDot kind={agentBOk ? "ok" : "err"}>
          B {agentBOk ? formatLatency(bLatency) : "failed"}
        </StatusDot>
      </div>
    </button>
  )
}

function HistoryDetail({
  detail,
  config,
}: {
  detail: HistoryDetailResponse
  config: ConfigResponse | null
}) {
  const latestJudge = detail.judges[0]?.result ?? null
  const agentNames = {
    a: detail.compare.agents.a.name,
    b: detail.compare.agents.b.name,
  }
  return (
    <div className="space-y-5">
      <Card>
        <CardContent className="space-y-3 p-[18px]">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="accent" className="font-mono">{detail.run_id}</Badge>
              <span className="font-mono text-[11px] text-arena-text-tertiary">
                {formatDateTime(detail.created_at)}
              </span>
              <Badge variant="neutral">Judge ×{detail.judges.length}</Badge>
              {latestJudge ? <WinnerBadge winner={latestJudge.winner} /> : null}
            </div>
          </div>
          <div className="rounded-arena border border-arena-border bg-arena-bg-subtle p-3 text-[13px] leading-relaxed text-arena-text-primary">
            {detail.input}
          </div>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[12.5px]">
            <KeyVal k="temperature" v={detail.temperature ?? "—"} />
            <KeyVal k="max_tokens" v={detail.max_tokens ?? "—"} />
            {detail.system ? (
              <KeyVal
                k="system"
                v={
                  <span className="truncate" title={detail.system}>
                    "{detail.system.length > 60 ? `${detail.system.slice(0, 60)}…` : detail.system}"
                  </span>
                }
              />
            ) : null}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-5 xl:grid-cols-2">
        <AgentResultPanel
          id="a"
          title="Agent A"
          result={detail.compare.agents.a}
          loading={false}
        />
        <AgentResultPanel
          id="b"
          title="Agent B"
          result={detail.compare.agents.b}
          loading={false}
        />
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

      {detail.judges.length > 1 ? <JudgeHistoryTable detail={detail} /> : null}

      <Card>
        <CardHeader>
          <CardTitle>原始数据</CardTitle>
        </CardHeader>
        <CardContent className="p-[18px]">
          <Tabs defaultValue="compare">
            <TabsList>
              <TabsTrigger value="compare">Compare</TabsTrigger>
              <TabsTrigger value="judges">Judges</TabsTrigger>
            </TabsList>
            <TabsContent value="compare" className="mt-3">
              <CodeBlock value={JSON.stringify(detail.compare, null, 2)} />
            </TabsContent>
            <TabsContent value="judges" className="mt-3">
              <CodeBlock
                value={detail.judges.length ? JSON.stringify(detail.judges, null, 2) : "尚无 Judge 结果"}
              />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}

function JudgeHistoryTable({ detail }: { detail: HistoryDetailResponse }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Judge 历史</CardTitle>
        <p className="mt-0.5 text-[11.5px] text-arena-text-tertiary">
          这条 run 被评估过 {detail.judges.length} 次
        </p>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-arena-bg-subtle text-[11px] font-semibold uppercase tracking-wider text-arena-text-tertiary">
              <tr>
                <th className="border-b border-arena-border px-3 py-2">#</th>
                <th className="border-b border-arena-border px-3 py-2">Judge 模型</th>
                <th className="border-b border-arena-border px-3 py-2">时间</th>
                <th className="border-b border-arena-border px-3 py-2">Winner</th>
                <th className="border-b border-arena-border px-3 py-2">摘要</th>
              </tr>
            </thead>
            <tbody>
              {detail.judges.map((record, index) => (
                <tr key={record.judge_id} className="border-b border-arena-border last:border-b-0">
                  <td className="px-3 py-2 font-mono text-[12px] text-arena-text-secondary">{index + 1}</td>
                  <td className="px-3 py-2 font-mono text-[12px] text-arena-text-primary">{record.model}</td>
                  <td className="px-3 py-2 font-mono text-[12px] text-arena-text-secondary">
                    {formatDateTime(record.created_at)}
                  </td>
                  <td className="px-3 py-2">
                    <WinnerBadge winner={record.result.winner} />
                  </td>
                  <td className="px-3 py-2 text-[12.5px] text-arena-text-secondary">
                    <span className="line-clamp-2">{record.result.summary || "—"}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
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
