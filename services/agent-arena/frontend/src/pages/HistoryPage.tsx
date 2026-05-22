import { AlertTriangle, History, ListRestart, Loader2, RefreshCcw } from "lucide-react"

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
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AgentResultPanel, EmptyBox, Metric } from "@/components/arena/AgentResultPanel"
import { CodeBlock } from "@/components/arena/CodeBlock"
import { JudgePanel } from "@/components/arena/JudgePanel"
import {
  formatDateTime,
  historyAgentName,
  historyAgentOk,
  winnerTextFromJudge,
} from "@/lib/formatters"
import type {
  ConfigResponse,
  HistoryDetailResponse,
  HistorySummary,
} from "@/lib/types"
import { cn } from "@/lib/utils"

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
