import { FormEvent, useEffect, useMemo, useState } from "react"
import { AlertTriangle, Download, KeyRound, LogOut, RefreshCcw } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { AppShell, PageBody, PageHeader } from "@/components/AppShell"
import {
  apiFetch,
  clearArenaAccessKey,
  getArenaAccessKey,
  isUnauthorizedError,
  readApiJson,
  setArenaAccessKey,
} from "@/lib/api"
import { normalizeError, reportFrontendLog } from "@/lib/frontendLogger"
import { samplePrompt } from "@/lib/formatters"
import type {
  CompareResponse,
  ConfigResponse,
  HistoryDetailResponse,
  HistoryListResponse,
  HistorySummary,
  JudgeResponse,
  ViewMode,
} from "@/lib/types"
import { BatchPage } from "@/BatchPage"
import { ArenaPage } from "@/pages/ArenaPage"
import { HistoryPage } from "@/pages/HistoryPage"

const PAGE_TITLE: Record<ViewMode, { title: string; badge?: string; subtitle: string }> = {
  arena: {
    title: "竞技场对比",
    badge: "A/B Compare",
    subtitle:
      "将同一条输入并发推送给两个 Agent，对比答案、过程事件与延迟，再交给 Judge 评分。",
  },
  batch: {
    title: "稳定性测试",
    badge: "Batch",
    subtitle: "同一输入多次迭代，统计成功率、延迟分布与 Judge 一致性。",
  },
  history: {
    title: "历史记录",
    badge: "Replay",
    subtitle: "所有对比与 Judge 评估结果，按时间倒序展示，支持回放。",
  },
}

function App() {
  const [view, setView] = useState<ViewMode>("arena")
  const [config, setConfig] = useState<ConfigResponse | null>(null)
  const [configError, setConfigError] = useState("")
  const [input, setInput] = useState(samplePrompt)
  const [system, setSystem] = useState("你是一个严谨、直接的技术助手。")
  const [temperature, setTemperature] = useState("0.2")
  const [maxTokens, setMaxTokens] = useState("1200")
  const [loading, setLoading] = useState(false)
  const [judgeLoading, setJudgeLoading] = useState(false)
  const [result, setResult] = useState<CompareResponse | null>(null)
  const [judge, setJudge] = useState<JudgeResponse | null>(null)
  const [error, setError] = useState("")
  const [historyItems, setHistoryItems] = useState<HistorySummary[]>([])
  const [historyDetail, setHistoryDetail] = useState<HistoryDetailResponse | null>(null)
  const [selectedHistoryRunId, setSelectedHistoryRunId] = useState("")
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyDetailLoading, setHistoryDetailLoading] = useState(false)
  const [historyError, setHistoryError] = useState("")
  const [accessKeyInput, setAccessKeyInput] = useState(() => getArenaAccessKey())
  const [needsAccessKey, setNeedsAccessKey] = useState(false)

  function markUnauthorized(message = "请输入 AgentArena access key。") {
    setNeedsAccessKey(true)
    setConfigError(message)
  }

  async function loadConfig() {
    setConfigError("")
    try {
      const response = await apiFetch("/config")
      setConfig(await readApiJson<ConfigResponse>(response))
      setNeedsAccessKey(false)
      setAccessKeyInput(getArenaAccessKey())
    } catch (err) {
      if (isUnauthorizedError(err)) {
        markUnauthorized()
        return
      }
      const normalized = normalizeError(err)
      reportFrontendLog({
        level: "error",
        source: "app.load_config",
        message: normalized.message,
        stack: normalized.stack,
      })
      setConfigError(normalized.message)
    }
  }

  function saveAccessKey(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setArenaAccessKey(accessKeyInput)
    setNeedsAccessKey(false)
    void loadConfig()
  }

  function openAccessKeyPrompt() {
    setAccessKeyInput(getArenaAccessKey())
    setNeedsAccessKey(true)
    setConfigError("")
  }

  function resetAccessKey() {
    clearArenaAccessKey()
    setAccessKeyInput("")
    setNeedsAccessKey(true)
    setConfig(null)
    setConfigError("请输入 AgentArena access key。")
  }

  useEffect(() => {
    void loadConfig()
  }, [])

  useEffect(() => {
    if (view === "history") {
      void loadHistory()
    }
  }, [view])

  const parsedTemperature = useMemo(() => {
    const value = Number.parseFloat(temperature)
    if (!Number.isFinite(value)) return 0.2
    return Math.min(2, Math.max(0, value))
  }, [temperature])

  const parsedMaxTokens = useMemo(() => {
    const value = Number.parseInt(maxTokens, 10)
    if (!Number.isFinite(value) || value <= 0) return undefined
    return value
  }, [maxTokens])

  const canJudge = Boolean(
    result &&
      !loading &&
      (result.agents.a.content.trim() || result.agents.b.content.trim()),
  )

  async function loadHistory() {
    setHistoryLoading(true)
    setHistoryError("")
    try {
      const response = await apiFetch("/history?limit=50")
      const history = await readApiJson<HistoryListResponse>(response)
      setHistoryItems(history.items)
      const nextRunId = selectedHistoryRunId || history.items[0]?.run_id || ""
      if (nextRunId) {
        setSelectedHistoryRunId(nextRunId)
        void loadHistoryDetail(nextRunId)
      } else {
        setHistoryDetail(null)
      }
    } catch (err) {
      if (isUnauthorizedError(err)) {
        markUnauthorized()
        setHistoryError("请输入 AgentArena access key。")
        return
      }
      const normalized = normalizeError(err)
      reportFrontendLog({
        level: "error",
        source: "app.history",
        message: normalized.message,
        stack: normalized.stack,
      })
      setHistoryError(normalized.message)
    } finally {
      setHistoryLoading(false)
    }
  }

  async function loadHistoryDetail(runId: string) {
    if (!runId) return
    setHistoryDetailLoading(true)
    setHistoryError("")
    try {
      const response = await apiFetch(`/history/${encodeURIComponent(runId)}`)
      setHistoryDetail(await readApiJson<HistoryDetailResponse>(response))
    } catch (err) {
      if (isUnauthorizedError(err)) {
        markUnauthorized()
        setHistoryError("请输入 AgentArena access key。")
        return
      }
      const normalized = normalizeError(err)
      reportFrontendLog({
        level: "error",
        source: "app.history_detail",
        message: normalized.message,
        stack: normalized.stack,
        payload: { run_id: runId },
      })
      setHistoryError(normalized.message)
    } finally {
      setHistoryDetailLoading(false)
    }
  }

  function selectHistoryRun(runId: string) {
    setSelectedHistoryRunId(runId)
    void loadHistoryDetail(runId)
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const trimmed = input.trim()
    if (!trimmed) {
      setError("请输入要对比的 prompt。")
      return
    }
    setError("")
    setLoading(true)
    setResult(null)
    setJudge(null)
    try {
      const response = await apiFetch("/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input: trimmed,
          system,
          temperature: parsedTemperature,
          max_tokens: parsedMaxTokens,
        }),
      })
      setResult(await readApiJson<CompareResponse>(response))
    } catch (err) {
      if (isUnauthorizedError(err)) {
        markUnauthorized()
        setError("请输入 AgentArena access key。")
        return
      }
      const normalized = normalizeError(err)
      reportFrontendLog({
        level: "error",
        source: "app.compare",
        message: normalized.message,
        stack: normalized.stack,
        payload: { input_length: input.length },
      })
      setError(normalized.message)
    } finally {
      setLoading(false)
    }
  }

  async function runJudge() {
    if (!result) return
    setJudgeLoading(true)
    setJudge(null)
    try {
      const response = await apiFetch("/judge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input,
          system,
          run_id: result.run_id,
          agent_a: result.agents.a,
          agent_b: result.agents.b,
        }),
      })
      setJudge(await readApiJson<JudgeResponse>(response))
    } catch (err) {
      if (isUnauthorizedError(err)) {
        markUnauthorized()
        setJudge({
          ok: false,
          winner: null,
          summary: "",
          answer_scores: {},
          process_scores: {},
          strengths: {},
          weaknesses: {},
          recommendations: {},
          latency_ms: null,
          error: "请输入 AgentArena access key。",
          raw: {},
        })
        return
      }
      const normalized = normalizeError(err)
      reportFrontendLog({
        level: "error",
        source: "app.judge",
        message: normalized.message,
        stack: normalized.stack,
      })
      setJudge({
        ok: false,
        winner: null,
        summary: "",
        answer_scores: {},
        process_scores: {},
        strengths: {},
        weaknesses: {},
        recommendations: {},
        latency_ms: null,
        error: normalized.message,
        raw: {},
      })
    } finally {
      setJudgeLoading(false)
    }
  }

  function clearAll() {
    setInput("")
    setSystem("")
    setResult(null)
    setJudge(null)
    setError("")
  }

  const agentCount = config ? 2 : 0
  const isConfigured = Boolean(config?.agents.a.configured && config?.agents.b.configured)
  const pageInfo = PAGE_TITLE[view]

  return (
    <AppShell
      view={view}
      onViewChange={setView}
      configured={isConfigured}
      agentCount={agentCount}
      historyCount={historyItems.length || undefined}
      onAccessKey={openAccessKeyPrompt}
    >
      <PageHeader
        title={pageInfo.title}
        badge={pageInfo.badge}
        subtitle={pageInfo.subtitle}
        actions={
          <>
            <Button type="button" variant="ghost" size="sm" onClick={loadConfig}>
              <RefreshCcw className="size-3.5" />
              刷新配置
            </Button>
            {view === "arena" ? (
              <Button type="button" variant="outline" size="sm" disabled>
                <Download className="size-3.5" />
                导出结果
              </Button>
            ) : null}
          </>
        }
      />
      <PageBody>
        {needsAccessKey ? (
          <Card className="mb-5">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <KeyRound className="size-4 text-arena-accent" />
                AgentArena 访问密钥
              </CardTitle>
              <CardDescription>
                后端设置了 ARENA_API_KEY，浏览器请求需要携带同一个 Bearer token。
              </CardDescription>
            </CardHeader>
            <CardContent className="p-[18px]">
              <form className="flex flex-col gap-3 sm:flex-row sm:items-end" onSubmit={saveAccessKey}>
                <div className="min-w-0 flex-1 space-y-2">
                  <Label htmlFor="arenaAccessKey">Access key</Label>
                  <Input
                    id="arenaAccessKey"
                    type="password"
                    value={accessKeyInput}
                    onChange={(event) => setAccessKeyInput(event.target.value)}
                    placeholder="ARENA_API_KEY"
                    autoComplete="off"
                  />
                </div>
                <div className="flex gap-2">
                  <Button type="submit">
                    <KeyRound className="size-4" />
                    保存
                  </Button>
                  <Button type="button" variant="outline" onClick={resetAccessKey}>
                    <LogOut className="size-4" />
                    清除
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>
        ) : null}

        {configError && !needsAccessKey ? (
          <Alert variant="destructive" className="mb-5">
            <AlertTriangle className="size-4" />
            <AlertTitle>配置读取失败</AlertTitle>
            <AlertDescription>{configError}</AlertDescription>
          </Alert>
        ) : null}

        {view === "batch" ? (
          <BatchPage config={config} />
        ) : view === "arena" ? (
          <ArenaPage
            input={input}
            setInput={setInput}
            system={system}
            setSystem={setSystem}
            temperature={temperature}
            setTemperature={setTemperature}
            maxTokens={maxTokens}
            setMaxTokens={setMaxTokens}
            parsedTemperature={parsedTemperature}
            parsedMaxTokens={parsedMaxTokens}
            loading={loading}
            judgeLoading={judgeLoading}
            result={result}
            judge={judge}
            error={error}
            canJudge={canJudge}
            config={config}
            onSubmit={submit}
            onJudge={runJudge}
            onClear={clearAll}
          />
        ) : (
          <HistoryPage
            items={historyItems}
            detail={historyDetail}
            selectedRunId={selectedHistoryRunId}
            loading={historyLoading}
            detailLoading={historyDetailLoading}
            error={historyError}
            config={config}
            onRefresh={loadHistory}
            onSelect={selectHistoryRun}
          />
        )}
      </PageBody>
    </AppShell>
  )
}

export default App
