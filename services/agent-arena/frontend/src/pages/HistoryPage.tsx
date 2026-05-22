import { useState } from "react"
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  GitBranch,
  Loader2,
  RefreshCcw,
  ShieldCheck,
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
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AgentIdBadge } from "@/components/AgentIdBadge"
import { AgentResultPanel, EmptyBox } from "@/components/arena/AgentResultPanel"
import { CodeBlock } from "@/components/arena/CodeBlock"
import { ConsistencyPanel } from "@/components/arena/ConsistencyPanel"
import { JudgePanel } from "@/components/arena/JudgePanel"
import { TraceModal } from "@/components/arena/TraceModal"
import { StatusDot } from "@/components/StatusDot"
import {
  formatDateTime,
  formatLatency,
  historyAgentOk,
} from "@/lib/formatters"
import { apiFetch, readApiJson } from "@/lib/api"
import { cn } from "@/lib/utils"
import type {
  AgentResult,
  BatchDetailItem,
  BatchDetailResponse,
  BatchHistorySummary,
  ConfigResponse,
  ConsistencyResponse,
  HistoryDetailResponse,
  HistoryItem,
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
  batchDetail,
  selectedRunId,
  selectedKind,
  loading,
  detailLoading,
  error,
  config,
  onRefresh,
  onSelect,
  onBatchRefresh,
}: {
  items: HistoryItem[]
  detail: HistoryDetailResponse | null
  batchDetail: BatchDetailResponse | null
  selectedRunId: string
  selectedKind: "arena" | "batch"
  loading: boolean
  detailLoading: boolean
  error: string
  config: ConfigResponse | null
  onRefresh: () => void
  onSelect: (runId: string, kind: "arena" | "batch") => void
  onBatchRefresh: () => Promise<void>
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
                {items.map((item) =>
                  item.kind === "arena" ? (
                    <ArenaListItem
                      item={item}
                      selected={item.run_id === selectedRunId && selectedKind === "arena"}
                      onClick={() => onSelect(item.run_id, "arena")}
                      key={`arena-${item.run_id}`}
                    />
                  ) : (
                    <BatchListItem
                      item={item}
                      selected={item.batch_id === selectedRunId && selectedKind === "batch"}
                      onClick={() => onSelect(item.batch_id, "batch")}
                      key={`batch-${item.batch_id}`}
                    />
                  ),
                )}
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
        ) : selectedKind === "arena" && detail ? (
          <ArenaDetail detail={detail} config={config} />
        ) : selectedKind === "batch" && batchDetail ? (
          <BatchDetail detail={batchDetail} onRefresh={onBatchRefresh} />
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

function KindBadge({ kind }: { kind: "arena" | "batch" }) {
  if (kind === "batch") {
    return (
      <span className="rounded-arena-sm bg-arena-info px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider text-white">
        BATCH
      </span>
    )
  }
  return (
    <span className="rounded-arena-sm bg-arena-accent px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider text-white">
      ARENA
    </span>
  )
}

function ArenaListItem({
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
        <KindBadge kind="arena" />
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

function BatchListItem({
  item,
  selected,
  onClick,
}: {
  item: BatchHistorySummary
  selected: boolean
  onClick: () => void
}) {
  const successPct =
    item.success_rate !== null && item.success_rate !== undefined
      ? `${(item.success_rate * 100).toFixed(1)}%`
      : "—"
  const latestConsistency = item.latest_consistency
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
        <KindBadge kind="batch" />
        <Badge variant="neutral">×{item.iterations}</Badge>
        <Badge variant="neutral">target {item.target}</Badge>
        {item.cancelled ? <Badge variant="warning">已取消</Badge> : null}
        {item.consistency_count > 0 ? (
          <Badge variant="info">稳定性 ×{item.consistency_count}</Badge>
        ) : null}
        <span className="ml-auto font-mono text-[10.5px] text-arena-text-tertiary">
          {relativeTime(item.created_at)}
        </span>
      </div>
      <div className="mt-2 line-clamp-2 text-[13px] leading-snug text-arena-text-primary">
        {item.input || "(无输入)"}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-3 text-[11.5px] text-arena-text-tertiary">
        <span className="font-mono">成功率 {successPct}</span>
        <span className="font-mono">runs {item.item_count}</span>
        {item.agent_summaries.map((s) => (
          <span key={s.agent_key} className="font-mono">
            {s.agent_key.toUpperCase()} P50 {formatLatency(s.latency_p50_ms)}
          </span>
        ))}
        {latestConsistency?.ok && latestConsistency.reports.length ? (
          <span className="inline-flex items-center gap-1 font-mono text-arena-info">
            <ShieldCheck className="size-3" />
            稳定 {Math.round(
              latestConsistency.reports.reduce(
                (sum, r) => sum + r.consistency_score,
                0,
              ) / latestConsistency.reports.length,
            )}
          </span>
        ) : null}
      </div>
    </button>
  )
}

function ArenaDetail({
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
              <KindBadge kind="arena" />
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
          input={detail.input}
        />
        <AgentResultPanel
          id="b"
          title="Agent B"
          result={detail.compare.agents.b}
          loading={false}
          input={detail.input}
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

function BatchDetail({
  detail,
  onRefresh,
}: {
  detail: BatchDetailResponse
  onRefresh: () => Promise<void>
}) {
  const latestConsistency = detail.consistency_results[0] ?? null
  return (
    <div className="space-y-5">
      <Card>
        <CardContent className="space-y-3 p-[18px]">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <KindBadge kind="batch" />
              <Badge variant="accent" className="font-mono">{detail.batch_id}</Badge>
              <span className="font-mono text-[11px] text-arena-text-tertiary">
                {formatDateTime(detail.created_at)}
              </span>
              <Badge variant="neutral">×{detail.iterations} runs</Badge>
              <Badge variant="neutral">target {detail.target}</Badge>
              <Badge variant="neutral">mode {detail.mode}</Badge>
              {detail.cancelled ? <Badge variant="warning">已取消</Badge> : null}
            </div>
          </div>
          {detail.items.length ? (
            <div className="rounded-arena border border-arena-border bg-arena-bg-subtle p-3 text-[13px] leading-relaxed text-arena-text-primary">
              {(() => {
                const req = detail.request as Record<string, unknown>
                const form = (req.form ?? {}) as { input?: string }
                const raw = (req.raw_body ?? {}) as { input?: string }
                return form.input || raw.input || "(无 input)"
              })()}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <BatchSummariesCards detail={detail} />

      <BatchConsistencyCard
        batchId={detail.batch_id}
        latest={latestConsistency}
        history={detail.consistency_results}
        onUpdated={onRefresh}
      />

      <BatchItemsTable detail={detail} />

      <Card>
        <CardHeader>
          <CardTitle>原始数据</CardTitle>
        </CardHeader>
        <CardContent className="p-[18px]">
          <Tabs defaultValue="summaries">
            <TabsList>
              <TabsTrigger value="summaries">Summaries</TabsTrigger>
              <TabsTrigger value="request">Request</TabsTrigger>
              <TabsTrigger value="consistency">Consistency</TabsTrigger>
            </TabsList>
            <TabsContent value="summaries" className="mt-3">
              <CodeBlock value={JSON.stringify(detail.summaries, null, 2)} />
            </TabsContent>
            <TabsContent value="request" className="mt-3">
              <CodeBlock value={JSON.stringify(detail.request, null, 2)} />
            </TabsContent>
            <TabsContent value="consistency" className="mt-3">
              <CodeBlock
                value={
                  detail.consistency_results.length
                    ? JSON.stringify(detail.consistency_results, null, 2)
                    : "尚未运行稳定性评估"
                }
              />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}

function BatchSummariesCards({ detail }: { detail: BatchDetailResponse }) {
  const entries = Object.entries(detail.summaries || {})
  if (!entries.length) {
    return null
  }
  return (
    <div className={cn("grid gap-5", entries.length > 1 ? "xl:grid-cols-2" : "")}>
      {entries.map(([key, raw]) => {
        const s = raw as Record<string, unknown>
        const total = (s.total as number) || 0
        const success = (s.success as number) || 0
        const successPct = total ? success / total : 0
        const successRateBadge =
          successPct >= 0.95 ? "success" : successPct >= 0.8 ? "warning" : "destructive"
        return (
          <Card key={key}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span className="font-mono text-xs uppercase tracking-wider text-arena-text-tertiary">
                  {key.toUpperCase()}
                </span>
                <span className="truncate">{String(s.agent_name || "")}</span>
                <Badge variant={successRateBadge} className="ml-auto">
                  成功率 {(successPct * 100).toFixed(1)}%
                </Badge>
              </CardTitle>
              <p className="mt-0.5 font-mono text-[11px] text-arena-text-tertiary">
                {String(s.agent_model || "—")}
              </p>
            </CardHeader>
            <CardContent className="p-[18px]">
              <div className="grid grid-cols-2 gap-x-5 gap-y-2 text-[12.5px]">
                <KeyVal k="total" v={total} />
                <KeyVal k="success" v={success} />
                <KeyVal k="P50" v={formatLatency(s.latency_p50_ms as number | null)} />
                <KeyVal k="P90" v={formatLatency(s.latency_p90_ms as number | null)} />
                <KeyVal
                  k="长度 avg"
                  v={
                    typeof s.content_len_avg === "number"
                      ? Math.round(s.content_len_avg)
                      : "—"
                  }
                />
                <KeyVal k="失败工具" v={(s.failed_tool_total as number | undefined) ?? 0} />
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}

function BatchConsistencyCard({
  batchId,
  latest,
  history,
  onUpdated,
}: {
  batchId: string
  latest: ConsistencyResponse | null
  history: ConsistencyResponse[]
  onUpdated: () => Promise<void>
}) {
  async function runEval() {
    const response = await apiFetch(`/batch/${encodeURIComponent(batchId)}/consistency`, {
      method: "POST",
    })
    await readApiJson(response)
    await onUpdated()
  }
  return (
    <ConsistencyPanel
      consistency={latest}
      history={history}
      onRun={runEval}
    />
  )
}

function batchItemToAgentResult(item: BatchDetailItem): AgentResult {
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

function batchInputFromRequest(detail: BatchDetailResponse): string | undefined {
  const req = detail.request as Record<string, unknown>
  const form = (req.form ?? {}) as { input?: string }
  const raw = (req.raw_body ?? {}) as { input?: string }
  return form.input || raw.input
}

function BatchItemsTable({ detail }: { detail: BatchDetailResponse }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [traceItem, setTraceItem] = useState<BatchDetailItem | null>(null)
  const userInput = batchInputFromRequest(detail)

  function rowKey(item: BatchDetailItem) {
    return `${item.agent_key}-${item.index}`
  }
  function toggle(item: BatchDetailItem) {
    setExpanded((prev) => {
      const next = new Set(prev)
      const key = rowKey(item)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>执行明细 ({detail.items.length})</CardTitle>
        <p className="mt-0.5 text-[11.5px] text-arena-text-tertiary">
          点击行展开完整 content，点击 <GitBranch className="inline size-3" /> 查看 Trace
        </p>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full table-fixed border-separate border-spacing-0 text-left">
            <colgroup>
              <col className="w-[48px]" />
              <col className="w-[56px]" />
              <col className="w-[180px]" />
              <col className="w-[88px]" />
              <col className="w-[78px]" />
              <col className="w-[64px]" />
              <col className="w-[88px]" />
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
                <th className="border-b border-arena-border px-2 py-2">预览</th>
              </tr>
            </thead>
            <tbody>
              {detail.items.length === 0 ? (
                <tr>
                  <td
                    colSpan={8}
                    className="px-2 py-10 text-center text-[13px] text-arena-text-tertiary"
                  >
                    无执行记录
                  </td>
                </tr>
              ) : (
                detail.items.map((item) => (
                  <HistoryItemRow
                    key={rowKey(item)}
                    item={item}
                    expanded={expanded.has(rowKey(item))}
                    onToggle={() => toggle(item)}
                    onOpenTrace={() => setTraceItem(item)}
                  />
                ))
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
      <TraceModal
        open={traceItem !== null}
        onOpenChange={(open) => {
          if (!open) setTraceItem(null)
        }}
        result={traceItem ? batchItemToAgentResult(traceItem) : null}
        title={
          traceItem
            ? `Trace · #${String(traceItem.index).padStart(3, "0")} · ${traceItem.agent_name}`
            : "Trace"
        }
        subtitle={traceItem ? traceItem.agent_model : ""}
        input={userInput}
      />
    </Card>
  )
}

function HistoryItemRow({
  item,
  expanded,
  onToggle,
  onOpenTrace,
}: {
  item: BatchDetailItem
  expanded: boolean
  onToggle: () => void
  onOpenTrace: () => void
}) {
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
          <StatusDot kind={item.ok ? "ok" : "err"}>
            {item.ok ? "ok" : "failed"}
          </StatusDot>
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
        <td className="truncate px-2 py-1.5 text-[12.5px] text-arena-text-secondary">
          <span className="line-clamp-1">
            {item.error
              ? `⚠ ${item.error}`
              : item.content
                ? item.content.slice(0, 200)
                : "(空)"}
          </span>
        </td>
      </tr>
      {expanded ? (
        <tr className="border-b border-arena-border bg-arena-bg-subtle">
          <td colSpan={8} className="px-4 py-3">
            <div className="space-y-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-arena-text-tertiary">
                Content ({item.content_length} chars)
              </div>
              <ScrollArea className="h-[260px] rounded-arena border border-arena-border bg-arena-bg-code p-3">
                <pre className="whitespace-pre-wrap break-words font-mono text-[12px] leading-5 text-slate-100">
                  {item.content || "(空)"}
                </pre>
              </ScrollArea>
              {item.error ? (
                <Alert variant="destructive">
                  <AlertTriangle className="size-4" />
                  <AlertTitle>错误</AlertTitle>
                  <AlertDescription className="whitespace-pre-wrap break-words">
                    {item.error}
                  </AlertDescription>
                </Alert>
              ) : null}
              <div className="text-[11px] text-arena-text-tertiary">
                点击行首 <GitBranch className="inline size-3" /> 图标查看完整 Trace 可视化
              </div>
            </div>
          </td>
        </tr>
      ) : null}
    </>
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
