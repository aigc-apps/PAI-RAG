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
import { AgentIdBadge } from "@/components/AgentIdBadge"
import { ScoreRow } from "@/components/ScoreBar"
import { EmptyBox } from "@/components/arena/AgentResultPanel"
import { formatLatency } from "@/lib/formatters"
import { cn } from "@/lib/utils"
import type { ConfigResponse, JudgeResponse } from "@/lib/types"

function coerceScore(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value)
    if (Number.isFinite(parsed)) return parsed
  }
  if (value && typeof value === "object") {
    for (const key of ["score", "value", "rating"]) {
      const v = (value as Record<string, unknown>)[key]
      const parsed = coerceScore(v)
      if (parsed !== null) return parsed
    }
  }
  return null
}

function extractScores(raw: Record<string, unknown>): Record<string, number> {
  const out: Record<string, number> = {}
  if (!raw || typeof raw !== "object") return out
  for (const [key, value] of Object.entries(raw)) {
    if (key === "a" || key === "b") continue
    const num = coerceScore(value)
    if (num !== null) out[key] = num
  }
  return out
}

function ScoreColumn({
  side,
  scores,
}: {
  side: "a" | "b"
  scores: Record<string, number>
}) {
  const entries = Object.entries(scores)
  return (
    <div className="space-y-1">
      <div className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-arena-text-secondary">
        <AgentIdBadge id={side} size="sm" />
        Agent {side.toUpperCase()} 维度得分
      </div>
      {entries.length === 0 ? (
        <p className="text-[12px] text-arena-text-tertiary">暂无评分</p>
      ) : (
        entries.map(([label, value]) => (
          <ScoreRow key={label} label={label} value={value} side={side} />
        ))
      )}
    </div>
  )
}

function ListSection({
  title,
  items,
  tone,
}: {
  title: string
  items?: string[]
  tone: "success" | "danger" | "info"
}) {
  const toneClass = {
    success: "text-arena-success",
    danger: "text-arena-danger",
    info: "text-arena-info",
  }[tone]
  return (
    <div className="space-y-1">
      <div className={cn("text-[11.5px] font-semibold", toneClass)}>{title}</div>
      {items && items.length > 0 ? (
        <ul className="list-disc space-y-0.5 pl-4 text-[12.5px] leading-relaxed text-arena-text-secondary marker:text-arena-text-tertiary">
          {items.map((item, index) => (
            <li key={`${title}-${index}`}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="text-[12px] text-arena-text-tertiary">暂无</p>
      )}
    </div>
  )
}

function JudgeSide({
  side,
  name,
  strengths,
  weaknesses,
  recommendations,
}: {
  side: "a" | "b"
  name: string
  strengths?: string[]
  weaknesses?: string[]
  recommendations?: string[]
}) {
  return (
    <div
      className={cn(
        "space-y-3 px-5 py-4",
        side === "a" ? "border-b border-arena-border xl:border-b-0 xl:border-r" : "",
      )}
    >
      <div className="flex items-center gap-1.5 text-[12.5px] font-semibold text-arena-text-primary">
        <AgentIdBadge id={side} size="sm" />
        {name}
      </div>
      <ListSection title="优势" items={strengths} tone="success" />
      <ListSection title="劣势" items={weaknesses} tone="danger" />
      <ListSection title="改进建议" items={recommendations} tone="info" />
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
  const nameA = agentNames?.a || config?.agents.a.name || "Agent A"
  const nameB = agentNames?.b || config?.agents.b.name || "Agent B"

  const winnerText =
    judge?.winner === "a"
      ? nameA
      : judge?.winner === "b"
        ? nameB
        : judge?.winner === "tie"
          ? "平局"
          : "未判定"

  const winnerLabel =
    judge?.winner === "a"
      ? "A"
      : judge?.winner === "b"
        ? "B"
        : judge?.winner === "tie"
          ? "TIE"
          : "—"

  const answerScores = judge ? judge.answer_scores : {}
  const aAnswer = extractScores(
    (answerScores?.a as Record<string, unknown>) || answerScores || {},
  )
  const bAnswer = extractScores(
    (answerScores?.b as Record<string, unknown>) || {},
  )
  const aProcess = extractScores(
    (judge?.process_scores?.a as Record<string, unknown>) || judge?.process_scores || {},
  )
  const bProcess = extractScores(
    (judge?.process_scores?.b as Record<string, unknown>) || {},
  )

  const aFinal = Object.keys(aAnswer).length > 0 ? aAnswer : aProcess
  const bFinal = Object.keys(bAnswer).length > 0 ? bAnswer : bProcess

  return (
    <Card>
      <CardHeader className="flex flex-row items-start gap-3">
        <div className="grid size-8 shrink-0 place-items-center rounded-arena bg-arena-accent-soft text-arena-accent-press">
          <Gavel className="size-4" />
        </div>
        <div className="min-w-0 flex-1">
          <CardTitle>Judge 评估</CardTitle>
          <CardDescription>
            使用 Judge 模型评估最终答案和公开过程事件。
          </CardDescription>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          <Badge variant={config?.judge.configured ? "success" : "warning"}>
            {config?.judge.configured ? "Judge 已配置" : "Judge 未配置"}
          </Badge>
          <Badge variant="neutral">{config?.judge.model || "未配置"}</Badge>
          {showAction ? (
            <Button
              type="button"
              onClick={onJudge}
              disabled={!canJudge || judgeLoading}
              size="sm"
            >
              {judgeLoading ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Scale className="size-4" />
              )}
              运行 Judge
            </Button>
          ) : null}
        </div>
      </CardHeader>
      <CardContent className="space-y-5 p-[18px]">
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
          <>
            <div className="flex flex-wrap items-center gap-3">
              <span
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-arena-sm px-2.5 py-1 font-mono text-[11px] font-bold uppercase tracking-wider text-white",
                  judge.winner === "tie"
                    ? "bg-arena-neutral"
                    : judge.winner === "b"
                      ? "bg-arena-success"
                      : "bg-arena-accent",
                )}
              >
                Winner · {winnerLabel}
              </span>
              <span className="text-[13px] text-arena-text-primary">{winnerText}</span>
              <Badge variant="neutral">耗时 {formatLatency(judge.latency_ms)}</Badge>
            </div>

            {judge.summary ? (
              <p className="text-[13px] leading-relaxed text-arena-text-secondary">
                {judge.summary}
              </p>
            ) : null}

            <div className="grid gap-6 xl:grid-cols-2">
              <ScoreColumn side="a" scores={aFinal} />
              <ScoreColumn side="b" scores={bFinal} />
            </div>

            <div className="overflow-hidden rounded-arena border border-arena-border bg-arena-bg-subtle xl:grid xl:grid-cols-2">
              <JudgeSide
                side="a"
                name={nameA}
                strengths={judge.strengths?.a}
                weaknesses={judge.weaknesses?.a}
                recommendations={judge.recommendations?.a}
              />
              <JudgeSide
                side="b"
                name={nameB}
                strengths={judge.strengths?.b}
                weaknesses={judge.weaknesses?.b}
                recommendations={judge.recommendations?.b}
              />
            </div>
          </>
        ) : (
          <EmptyBox text={showAction ? "生成两个答案后点击 \"运行 Judge\" 开始裁判评估" : "这次 PK 还没有裁判评估"} />
        )}
      </CardContent>
    </Card>
  )
}
