import { AlertTriangle, Gavel, Loader2, Scale } from "lucide-react"

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
import { Skeleton } from "@/components/ui/skeleton"
import { CodeBlock } from "@/components/arena/CodeBlock"
import { EmptyBox } from "@/components/arena/AgentResultPanel"
import { formatLatency } from "@/lib/formatters"
import type { ConfigResponse, JudgeResponse } from "@/lib/types"

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

export function JudgePanel({
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
