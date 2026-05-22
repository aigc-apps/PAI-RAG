import { useState } from "react"
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Loader2,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
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
import { AgentIdBadge } from "@/components/AgentIdBadge"
import { Callout } from "@/components/Callout"
import { ScoreBar } from "@/components/ScoreBar"
import { StatusDot } from "@/components/StatusDot"
import { EmptyBox } from "@/components/arena/AgentResultPanel"
import { formatDateTime, formatLatency } from "@/lib/formatters"
import { cn } from "@/lib/utils"
import { normalizeError, reportFrontendLog } from "@/lib/frontendLogger"
import type {
  ConsistencyAgentReport,
  ConsistencyIssue,
  ConsistencyResponse,
} from "@/lib/types"

const SEVERITY_STYLES: Record<
  ConsistencyIssue["severity"],
  { badge: "destructive" | "warning" | "neutral"; label: string }
> = {
  high: { badge: "destructive", label: "高" },
  medium: { badge: "warning", label: "中" },
  low: { badge: "neutral", label: "低" },
}

export function ConsistencyPanel({
  consistency,
  history,
  onRun,
}: {
  consistency: ConsistencyResponse | null
  history: ConsistencyResponse[]
  onRun: () => Promise<void>
}) {
  const [running, setRunning] = useState(false)
  const [error, setError] = useState("")
  const [showHistory, setShowHistory] = useState(false)

  async function trigger() {
    setRunning(true)
    setError("")
    try {
      await onRun()
    } catch (err) {
      const normalized = normalizeError(err)
      reportFrontendLog({
        level: "error",
        source: "consistency.run",
        message: normalized.message,
        stack: normalized.stack,
      })
      setError(normalized.message)
    } finally {
      setRunning(false)
    }
  }

  const olderRuns = history.length > 1 ? history.slice(1) : []
  const reports = consistency?.reports ?? []
  const avgScore =
    reports.length > 0
      ? Math.round(
          reports.reduce((sum, r) => sum + r.consistency_score, 0) / reports.length,
        )
      : null

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div>
          <CardTitle className="flex items-center gap-2">
            <ShieldCheck className="size-4 text-arena-info" />
            稳定性评估
            {avgScore !== null ? (
              <Badge variant="info" className="font-mono">
                平均 {avgScore}/100
              </Badge>
            ) : null}
            {consistency ? (
              <Badge variant="neutral">{formatLatency(consistency.latency_ms)}</Badge>
            ) : null}
          </CardTitle>
          <CardDescription>
            交给 Judge 模型判断多次执行结果是否稳定、是否存在明显错误，并给出修复建议。
          </CardDescription>
        </div>
        <Button type="button" size="sm" onClick={trigger} disabled={running}>
          {running ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : (
            <Sparkles className="size-3.5" />
          )}
          {consistency ? "重新评估" : "运行稳定性评估"}
        </Button>
      </CardHeader>
      <CardContent className="space-y-4 p-[18px] pt-0">
        {error ? (
          <Alert variant="destructive">
            <AlertTriangle className="size-4" />
            <AlertTitle>稳定性评估失败</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        {!consistency ? (
          <EmptyBox text="尚未运行稳定性评估。点击右上角按钮开始第一次评估。" />
        ) : !consistency.ok ? (
          <Alert variant="destructive">
            <AlertTriangle className="size-4" />
            <AlertTitle>评估未完成</AlertTitle>
            <AlertDescription>{consistency.error || "Judge 未返回有效结果"}</AlertDescription>
          </Alert>
        ) : reports.length === 0 ? (
          <EmptyBox text="评估返回空结果，请重试。" />
        ) : (
          <div className="space-y-4">
            <div
              className={cn(
                "grid gap-4",
                reports.length > 1 ? "xl:grid-cols-2" : "",
              )}
            >
              {reports.map((report) => (
                <AgentReportCard key={report.agent_key} report={report} />
              ))}
            </div>
          </div>
        )}

        {olderRuns.length > 0 ? (
          <div className="rounded-arena border border-arena-border bg-arena-bg-subtle">
            <button
              type="button"
              onClick={() => setShowHistory((v) => !v)}
              className="flex w-full items-center justify-between px-3 py-2 text-left text-[12px] font-semibold text-arena-text-secondary hover:text-arena-text-primary"
            >
              <span className="inline-flex items-center gap-1.5">
                {showHistory ? (
                  <ChevronDown className="size-3.5" />
                ) : (
                  <ChevronRight className="size-3.5" />
                )}
                评估历史 ×{history.length}
              </span>
              <span className="font-mono text-[11px] text-arena-text-tertiary">
                显示 {olderRuns.length} 条更早的评估
              </span>
            </button>
            {showHistory ? (
              <div className="divide-y divide-arena-border border-t border-arena-border">
                {olderRuns.map((run, idx) => (
                  <HistoryRow key={`${run.created_at}-${idx}`} run={run} />
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}

function AgentReportCard({ report }: { report: ConsistencyAgentReport }) {
  const score = Math.max(0, Math.min(100, Math.round(report.consistency_score)))
  return (
    <div className="rounded-arena border border-arena-border bg-white p-3.5 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <AgentIdBadge id={report.agent_key} size="sm" />
          <span className="text-[13px] font-semibold text-arena-text-primary">
            {report.agent_name || `Agent ${report.agent_key.toUpperCase()}`}
          </span>
          <Badge variant="neutral" className="font-mono">
            样本 {report.samples_evaluated}
          </Badge>
        </div>
        {report.stable ? (
          <StatusDot kind="ok">稳定</StatusDot>
        ) : (
          <StatusDot kind="warn">不稳定</StatusDot>
        )}
      </div>

      <div className="space-y-1">
        <div className="flex items-center justify-between text-[11px] text-arena-text-tertiary">
          <span>一致性分数</span>
          <span className="font-mono text-arena-text-primary">{score}/100</span>
        </div>
        <ScoreBar value={score} side={report.agent_key} />
      </div>

      {report.summary ? (
        <p className="whitespace-pre-wrap text-[12.5px] leading-relaxed text-arena-text-secondary">
          {report.summary}
        </p>
      ) : null}

      {report.issues.length > 0 ? (
        <div>
          <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-arena-text-secondary">
            <ShieldAlert className="size-3" />
            问题 ({report.issues.length})
          </div>
          <ul className="space-y-1.5">
            {report.issues.map((issue, idx) => {
              const severity = SEVERITY_STYLES[issue.severity] ?? SEVERITY_STYLES.low
              return (
                <li
                  key={`${issue.index}-${idx}`}
                  className="flex items-start gap-2 rounded-arena-sm border border-arena-border bg-arena-bg-subtle px-2.5 py-1.5 text-[12px]"
                >
                  <Badge variant={severity.badge} className="shrink-0">
                    {severity.label}
                  </Badge>
                  <span className="font-mono text-[11px] text-arena-text-tertiary">
                    #{String(issue.index).padStart(3, "0")}
                  </span>
                  <span className="flex-1 text-arena-text-primary">{issue.problem}</span>
                </li>
              )
            })}
          </ul>
        </div>
      ) : null}

      {report.suggestions.length > 0 ? (
        <Callout
          variant={report.stable ? "info" : "warning"}
          title="修复建议"
          icon={<Sparkles className="size-3.5" />}
        >
          <ul className="list-disc space-y-1 pl-4">
            {report.suggestions.map((s, idx) => (
              <li key={idx} className="leading-relaxed">
                {s}
              </li>
            ))}
          </ul>
        </Callout>
      ) : report.issues.length === 0 && report.stable ? (
        <div className="flex items-center gap-1.5 text-[12px] text-arena-success">
          <CheckCircle2 className="size-3.5" />
          未发现明显问题
        </div>
      ) : null}
    </div>
  )
}

function HistoryRow({ run }: { run: ConsistencyResponse }) {
  const avgScore = run.reports.length
    ? Math.round(
        run.reports.reduce((sum, r) => sum + r.consistency_score, 0) /
          run.reports.length,
      )
    : null
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2 text-[12px]">
      <span className="font-mono text-[11px] text-arena-text-tertiary">
        {formatDateTime(run.created_at)}
      </span>
      <span className="font-mono text-[11px] text-arena-text-secondary">{run.model}</span>
      {avgScore !== null ? (
        <Badge variant="info" className="font-mono">
          平均 {avgScore}/100
        </Badge>
      ) : null}
      {run.reports.map((r) => (
        <span
          key={r.agent_key}
          className="inline-flex items-center gap-1 font-mono text-[11px] text-arena-text-secondary"
        >
          {r.agent_key.toUpperCase()} {Math.round(r.consistency_score)}/100
          {r.stable ? null : (
            <span className="text-arena-warning">·{r.issues.length} 问题</span>
          )}
        </span>
      ))}
      {run.error ? (
        <span className="text-arena-danger">{run.error}</span>
      ) : null}
    </div>
  )
}
