import type {
  HistorySummary,
  JudgeResponse,
  PublicAgentConfig,
} from "@/lib/types"

export function formatLatency(value: number | null | undefined) {
  if (value === null || value === undefined) return "未返回"
  if (value < 1000) return `${value} ms`
  return `${(value / 1000).toFixed(1)} s`
}

export function formatDateTime(value: string | null | undefined) {
  if (!value) return "未知时间"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString("zh-CN", { hour12: false })
}

export function modelLabel(agent?: PublicAgentConfig) {
  if (!agent) return "未加载配置"
  return agent.model || "未配置模型"
}

export function historyAgentName(summary: Record<string, unknown>, fallback: string) {
  return typeof summary.name === "string" && summary.name ? summary.name : fallback
}

export function historyAgentModel(summary: Record<string, unknown>) {
  return typeof summary.model === "string" && summary.model ? summary.model : "未记录模型"
}

export function historyAgentOk(summary: Record<string, unknown>) {
  return summary.ok === true
}

export function winnerTextFromJudge(
  judge: JudgeResponse | null | undefined,
  item: HistorySummary,
) {
  if (!judge?.winner) return "未评估"
  if (judge.winner === "tie") return "平局"
  if (judge.winner === "a") return historyAgentName(item.agent_a, "Agent A")
  if (judge.winner === "b") return historyAgentName(item.agent_b, "Agent B")
  return "未判定"
}

export const samplePrompt =
  "请用三个要点说明：如果要给一个项目加入长期记忆能力，最重要的设计取舍是什么？"
