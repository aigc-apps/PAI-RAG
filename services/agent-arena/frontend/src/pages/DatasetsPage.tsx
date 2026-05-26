import { FormEvent, useEffect, useState } from "react"
import {
  AlertTriangle,
  ArrowLeft,
  ChevronRight,
  Database,
  FileText,
  History as HistoryIcon,
  Pencil,
  Play,
  Plus,
  RefreshCcw,
  Sparkles,
  Trash2,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { apiFetch, isUnauthorizedError, readApiJson } from "@/lib/api"
import { normalizeError } from "@/lib/frontendLogger"
import { formatDateTime } from "@/lib/formatters"
import type {
  ArenaRunCandidate,
  Dataset,
  DatasetCase,
  DatasetRun,
  DatasetRunDetail,
  DatasetWithChildren,
} from "@/lib/types"

type CaseFormState = {
  id: string | null
  query: string
  system_prompt: string
  expected_answer: string
  tags: string
}

const EMPTY_CASE_FORM: CaseFormState = {
  id: null,
  query: "",
  system_prompt: "",
  expected_answer: "",
  tags: "",
}

export function DatasetsPage({ onUnauthorized }: { onUnauthorized?: () => void }) {
  const [selected, setSelected] = useState<string | null>(null)

  if (selected) {
    return (
      <DatasetDetail
        datasetId={selected}
        onBack={() => setSelected(null)}
        onUnauthorized={onUnauthorized}
      />
    )
  }
  return <DatasetList onSelect={(id) => setSelected(id)} onUnauthorized={onUnauthorized} />
}

// ---------- List ------------------------------------------------------------

function DatasetList({
  onSelect,
  onUnauthorized,
}: {
  onSelect: (id: string) => void
  onUnauthorized?: () => void
}) {
  const [items, setItems] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [dialogOpen, setDialogOpen] = useState(false)
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [formError, setFormError] = useState("")

  async function load() {
    setLoading(true)
    setError("")
    try {
      const response = await apiFetch("/datasets")
      const data = await readApiJson<{ datasets: Dataset[] }>(response)
      setItems(data.datasets)
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
      setError(normalizeError(err).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  function openCreate() {
    setName("")
    setDescription("")
    setFormError("")
    setDialogOpen(true)
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!name.trim()) {
      setFormError("评测集名称不能为空。")
      return
    }
    setSubmitting(true)
    setFormError("")
    try {
      const response = await apiFetch("/datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description }),
      })
      await readApiJson(response)
      setDialogOpen(false)
      await load()
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
      setFormError(normalizeError(err).message)
    } finally {
      setSubmitting(false)
    }
  }

  async function remove(ds: Dataset) {
    if (!window.confirm(`删除评测集「${ds.name}」？所有用例和评测记录将一起删除。`)) return
    try {
      const response = await apiFetch(`/datasets/${ds.id}`, { method: "DELETE" })
      await readApiJson(response)
      await load()
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
      setError(normalizeError(err).message)
    }
  }

  return (
    <div className="space-y-5">
      {error ? (
        <Alert variant="destructive">
          <AlertTriangle className="size-4" />
          <AlertTitle>读取失败</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Database className="size-4 text-arena-accent" />
              评测集
            </CardTitle>
            <p className="mt-1 text-[12px] text-arena-text-tertiary">
              每个评测集 = 一组 query + 可选黄金答案；运行后由 Judge 评 A/B 输出。
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button type="button" variant="ghost" size="sm" onClick={load}>
              <RefreshCcw className="size-3.5" />
              刷新
            </Button>
            <Button type="button" size="sm" onClick={openCreate}>
              <Plus className="size-3.5" />
              新建评测集
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="px-6 py-10 text-center text-[13px] text-arena-text-tertiary">加载中…</div>
          ) : items.length === 0 ? (
            <div className="px-6 py-10 text-center text-[13px] text-arena-text-tertiary">
              暂无评测集。点击右上「新建评测集」开始。
            </div>
          ) : (
            <div className="divide-y divide-arena-border">
              {items.map((ds) => (
                <button
                  key={ds.id}
                  type="button"
                  onClick={() => onSelect(ds.id)}
                  className="flex w-full items-center justify-between gap-3 px-6 py-3.5 text-left hover:bg-arena-bg-subtle"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-arena-text-primary">{ds.name}</span>
                      <Badge variant="outline" className="font-mono text-[10px]">
                        {ds.case_count ?? 0} cases
                      </Badge>
                    </div>
                    {ds.description ? (
                      <div className="mt-0.5 text-[11px] text-arena-text-tertiary">
                        {ds.description}
                      </div>
                    ) : null}
                    <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-arena-text-tertiary">
                      <span>更新于 {formatDateTime(ds.updated_at)}</span>
                      {ds.last_run ? (
                        <>
                          <span>·</span>
                          <span>
                            上次评测 {formatDateTime(ds.last_run.created_at)} ·{" "}
                            <Badge variant="outline" className="ml-1 font-mono text-[10px] uppercase">
                              {ds.last_run.status}
                            </Badge>
                          </span>
                        </>
                      ) : null}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        void remove(ds)
                      }}
                    >
                      <Trash2 className="size-3.5 text-red-600" />
                    </Button>
                    <ChevronRight className="size-4 text-arena-text-tertiary" />
                  </div>
                </button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="w-[min(560px,94vw)]">
          <form onSubmit={submit} className="flex flex-col gap-4 p-6">
            <div>
              <DialogTitle>新建评测集</DialogTitle>
              <DialogDescription>
                给评测集起个能区分场景的名字，比如「金融 FAQ v1」「Code-Review-2026Q1」。
              </DialogDescription>
            </div>
            <div className="space-y-1.5">
              <Label className="text-[12px]">名称</Label>
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="金融 FAQ v1" />
            </div>
            <div className="space-y-1.5">
              <Label className="text-[12px]">说明（可选）</Label>
              <Textarea
                rows={2}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="一行描述这个评测集覆盖的场景或目的"
              />
            </div>
            {formError ? <div className="text-[12px] text-red-600">{formError}</div> : null}
            <div className="flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={() => setDialogOpen(false)}>
                取消
              </Button>
              <Button type="submit" disabled={submitting}>
                {submitting ? "创建中…" : "创建"}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  )
}

// ---------- Detail ----------------------------------------------------------

function DatasetDetail({
  datasetId,
  onBack,
  onUnauthorized,
}: {
  datasetId: string
  onBack: () => void
  onUnauthorized?: () => void
}) {
  const [detail, setDetail] = useState<DatasetWithChildren | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [tab, setTab] = useState<"cases" | "candidates" | "runs">("cases")

  const [caseDialogOpen, setCaseDialogOpen] = useState(false)
  const [caseForm, setCaseForm] = useState<CaseFormState>(EMPTY_CASE_FORM)
  const [caseSubmitting, setCaseSubmitting] = useState(false)
  const [caseError, setCaseError] = useState("")

  const [runStarting, setRunStarting] = useState(false)
  const [runMessage, setRunMessage] = useState("")

  async function load() {
    setLoading(true)
    setError("")
    try {
      const response = await apiFetch(`/datasets/${datasetId}`)
      const data = await readApiJson<DatasetWithChildren>(response)
      setDetail(data)
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
      setError(normalizeError(err).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [datasetId])

  function openCreateCase() {
    setCaseForm(EMPTY_CASE_FORM)
    setCaseError("")
    setCaseDialogOpen(true)
  }

  function openEditCase(c: DatasetCase) {
    setCaseForm({
      id: c.id,
      query: c.query,
      system_prompt: c.system_prompt,
      expected_answer: c.expected_answer,
      tags: c.tags.join(", "),
    })
    setCaseError("")
    setCaseDialogOpen(true)
  }

  async function submitCase(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!caseForm.query.trim()) {
      setCaseError("Query 不能为空。")
      return
    }
    setCaseSubmitting(true)
    setCaseError("")
    const tags = caseForm.tags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean)
    try {
      const path = caseForm.id ? `/datasets/cases/${caseForm.id}` : `/datasets/${datasetId}/cases`
      const method = caseForm.id ? "PATCH" : "POST"
      const response = await apiFetch(path, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: caseForm.query,
          system_prompt: caseForm.system_prompt,
          expected_answer: caseForm.expected_answer,
          tags,
        }),
      })
      await readApiJson(response)
      setCaseDialogOpen(false)
      await load()
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
      setCaseError(normalizeError(err).message)
    } finally {
      setCaseSubmitting(false)
    }
  }

  async function deleteCase(c: DatasetCase) {
    if (!window.confirm(`删除用例？\n\n${c.query.slice(0, 80)}`)) return
    try {
      const response = await apiFetch(`/datasets/cases/${c.id}`, { method: "DELETE" })
      await readApiJson(response)
      await load()
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
      setError(normalizeError(err).message)
    }
  }

  async function startRun(judgeEach: boolean) {
    if (!detail || detail.cases.length === 0) {
      setRunMessage("当前评测集没有用例，先去 Cases 标签添加。")
      return
    }
    if (!window.confirm(`将对全部 ${detail.cases.length} 个用例并行调用 A/B，${judgeEach ? "且每条调用 Judge 评分" : "不跑 Judge"}。继续？`)) return
    setRunStarting(true)
    setRunMessage("")
    try {
      const response = await apiFetch(`/datasets/${datasetId}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ judge_each: judgeEach }),
      })
      const run = await readApiJson<DatasetRun>(response)
      setRunMessage(`已开始评测 · run_id ${run.id} · 状态 ${run.status}`)
      setTab("runs")
      await load()
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
      setRunMessage(`启动失败：${normalizeError(err).message}`)
    } finally {
      setRunStarting(false)
    }
  }

  if (loading || !detail) {
    return (
      <div className="space-y-5">
        <Button type="button" variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="size-3.5" />
          返回评测集列表
        </Button>
        <div className="py-10 text-center text-[13px] text-arena-text-tertiary">
          {error || "加载中…"}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Button type="button" variant="ghost" size="sm" onClick={onBack}>
            <ArrowLeft className="size-3.5" />
            评测集列表
          </Button>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-base font-semibold text-arena-text-primary">{detail.name}</span>
              <Badge variant="outline" className="font-mono text-[10px]">
                {detail.cases.length} cases
              </Badge>
            </div>
            {detail.description ? (
              <div className="mt-0.5 text-[12px] text-arena-text-tertiary">{detail.description}</div>
            ) : null}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={runStarting}
            onClick={() => startRun(false)}
          >
            <Play className="size-3.5" />
            仅跑 A/B
          </Button>
          <Button type="button" size="sm" disabled={runStarting} onClick={() => startRun(true)}>
            <Sparkles className="size-3.5" />
            {runStarting ? "启动中…" : "跑评测 (含 Judge)"}
          </Button>
        </div>
      </div>

      {error ? (
        <Alert variant="destructive">
          <AlertTriangle className="size-4" />
          <AlertTitle>读取失败</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      {runMessage ? (
        <Alert>
          <AlertDescription>{runMessage}</AlertDescription>
        </Alert>
      ) : null}

      <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)}>
        <TabsList>
          <TabsTrigger value="cases">
            <FileText className="size-3.5" />
            Cases · {detail.cases.length}
          </TabsTrigger>
          <TabsTrigger value="candidates">
            <Sparkles className="size-3.5" />
            Candidates
          </TabsTrigger>
          <TabsTrigger value="runs">
            <HistoryIcon className="size-3.5" />
            Runs · {detail.runs.length}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="cases" className="mt-4">
          <CasesPanel
            cases={detail.cases}
            onCreate={openCreateCase}
            onEdit={openEditCase}
            onDelete={deleteCase}
          />
        </TabsContent>

        <TabsContent value="candidates" className="mt-4">
          <CandidatesPanel datasetId={datasetId} onHarvested={load} onUnauthorized={onUnauthorized} />
        </TabsContent>

        <TabsContent value="runs" className="mt-4">
          <RunsPanel runs={detail.runs} onUnauthorized={onUnauthorized} />
        </TabsContent>
      </Tabs>

      <Dialog open={caseDialogOpen} onOpenChange={setCaseDialogOpen}>
        <DialogContent className="w-[min(680px,96vw)]">
          <form onSubmit={submitCase} className="flex flex-col gap-4 p-6">
            <div>
              <DialogTitle>{caseForm.id ? "编辑用例" : "新增用例"}</DialogTitle>
              <DialogDescription>
                Query 必填；Expected answer 可留空 — Judge 会同时评判答案与过程合理性。
              </DialogDescription>
            </div>
            <div className="space-y-1.5">
              <Label className="text-[12px]">Query</Label>
              <Textarea
                rows={3}
                value={caseForm.query}
                onChange={(e) => setCaseForm({ ...caseForm, query: e.target.value })}
                placeholder="用户的问题或指令"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-[12px]">System prompt（可选）</Label>
              <Textarea
                rows={2}
                value={caseForm.system_prompt}
                onChange={(e) => setCaseForm({ ...caseForm, system_prompt: e.target.value })}
                placeholder="为这个用例临时覆盖 system；为空则用全局默认"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-[12px]">黄金答案（可选）</Label>
              <Textarea
                rows={3}
                value={caseForm.expected_answer}
                onChange={(e) => setCaseForm({ ...caseForm, expected_answer: e.target.value })}
                placeholder="提供后 Judge 可对照打分；不提供则仅评过程"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-[12px]">Tags（逗号分隔）</Label>
              <Input
                value={caseForm.tags}
                onChange={(e) => setCaseForm({ ...caseForm, tags: e.target.value })}
                placeholder="hard, multi-turn, regression"
              />
            </div>
            {caseError ? <div className="text-[12px] text-red-600">{caseError}</div> : null}
            <div className="flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={() => setCaseDialogOpen(false)}>
                取消
              </Button>
              <Button type="submit" disabled={caseSubmitting}>
                {caseSubmitting ? "保存中…" : "保存"}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  )
}

// ---------- Cases panel -----------------------------------------------------

function CasesPanel({
  cases,
  onCreate,
  onEdit,
  onDelete,
}: {
  cases: DatasetCase[]
  onCreate: () => void
  onEdit: (c: DatasetCase) => void
  onDelete: (c: DatasetCase) => void
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle>用例</CardTitle>
        <Button type="button" size="sm" onClick={onCreate}>
          <Plus className="size-3.5" />
          新增用例
        </Button>
      </CardHeader>
      <CardContent className="p-0">
        {cases.length === 0 ? (
          <div className="px-6 py-10 text-center text-[13px] text-arena-text-tertiary">
            暂无用例。点击「新增用例」或在 Candidates 标签里从历史记录采集。
          </div>
        ) : (
          <div className="divide-y divide-arena-border">
            {cases.map((c, i) => (
              <div key={c.id} className="px-6 py-3.5">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1 text-[13px]">
                    <div className="flex items-center gap-2 text-[11px] text-arena-text-tertiary">
                      <span className="font-mono">#{i + 1}</span>
                      {c.source_run_id ? (
                        <Badge variant="outline" className="font-mono text-[10px]">harvested</Badge>
                      ) : null}
                      {c.tags.map((t) => (
                        <Badge key={t} variant="outline" className="text-[10px]">
                          {t}
                        </Badge>
                      ))}
                    </div>
                    <div className="mt-1 whitespace-pre-wrap text-arena-text-primary">{c.query}</div>
                    {c.expected_answer ? (
                      <div className="mt-2 rounded-arena border border-arena-border bg-arena-bg-subtle px-3 py-2 text-[12px] text-arena-text-secondary">
                        <span className="mr-2 font-semibold uppercase tracking-wider text-arena-text-tertiary">
                          黄金答案
                        </span>
                        <span className="whitespace-pre-wrap">{c.expected_answer}</span>
                      </div>
                    ) : null}
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Button type="button" variant="ghost" size="sm" onClick={() => onEdit(c)}>
                      <Pencil className="size-3.5" />
                    </Button>
                    <Button type="button" variant="ghost" size="sm" onClick={() => onDelete(c)}>
                      <Trash2 className="size-3.5 text-red-600" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ---------- Candidates panel ------------------------------------------------

function CandidatesPanel({
  datasetId,
  onHarvested,
  onUnauthorized,
}: {
  datasetId: string
  onHarvested: () => void
  onUnauthorized?: () => void
}) {
  const [candidates, setCandidates] = useState<ArenaRunCandidate[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [savingId, setSavingId] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    setError("")
    try {
      const response = await apiFetch("/arena-runs/candidates?limit=30")
      const data = await readApiJson<{ candidates: ArenaRunCandidate[] }>(response)
      setCandidates(data.candidates)
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
      setError(normalizeError(err).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  async function harvest(run: ArenaRunCandidate) {
    setSavingId(run.run_id)
    try {
      const response = await apiFetch(`/arena-runs/${run.run_id}/save-to-dataset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: datasetId }),
      })
      await readApiJson(response)
      onHarvested()
      await load()
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
      setError(normalizeError(err).message)
    } finally {
      setSavingId(null)
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0">
        <div>
          <CardTitle>从历史 Arena 运行采集</CardTitle>
          <p className="mt-1 text-[12px] text-arena-text-tertiary">
            最近 30 条尚未被任何评测集吸纳的对比记录。点击「加入评测集」就把它存为本评测集的用例。
          </p>
        </div>
        <Button type="button" variant="ghost" size="sm" onClick={load}>
          <RefreshCcw className="size-3.5" />
          刷新
        </Button>
      </CardHeader>
      <CardContent className="p-0">
        {error ? (
          <div className="px-6 py-3 text-[12px] text-red-600">{error}</div>
        ) : null}
        {loading ? (
          <div className="px-6 py-10 text-center text-[13px] text-arena-text-tertiary">加载中…</div>
        ) : candidates.length === 0 ? (
          <div className="px-6 py-10 text-center text-[13px] text-arena-text-tertiary">
            没有可采集的候选运行（全部已被采集，或暂无历史 Arena 记录）。
          </div>
        ) : (
          <div className="divide-y divide-arena-border">
            {candidates.map((run) => (
              <div key={run.run_id} className="flex items-start gap-3 px-6 py-3.5">
                <div className="min-w-0 flex-1">
                  <div className="text-[11px] text-arena-text-tertiary">
                    {formatDateTime(run.created_at)} · {run.agent_a_name} vs {run.agent_b_name}
                  </div>
                  <div className="mt-1 line-clamp-2 whitespace-pre-wrap text-[13px] text-arena-text-primary">
                    {run.input}
                  </div>
                </div>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  disabled={savingId === run.run_id}
                  onClick={() => harvest(run)}
                >
                  <Plus className="size-3.5" />
                  {savingId === run.run_id ? "保存中…" : "加入"}
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ---------- Runs panel ------------------------------------------------------

function RunsPanel({
  runs,
  onUnauthorized,
}: {
  runs: DatasetRun[]
  onUnauthorized?: () => void
}) {
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [detail, setDetail] = useState<DatasetRunDetail | null>(null)
  const [loading, setLoading] = useState(false)

  async function loadDetail(runId: string) {
    setLoading(true)
    try {
      const response = await apiFetch(`/datasets/runs/${runId}`)
      setDetail(await readApiJson<DatasetRunDetail>(response))
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!selectedRun && runs[0]) {
      setSelectedRun(runs[0].id)
    }
  }, [runs])

  useEffect(() => {
    if (selectedRun) void loadDetail(selectedRun)
  }, [selectedRun])

  if (runs.length === 0) {
    return (
      <Card>
        <CardContent className="px-6 py-10 text-center text-[13px] text-arena-text-tertiary">
          暂无评测运行。点击页面右上「跑评测」开始第一次评测。
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-[13px]">评测运行</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="divide-y divide-arena-border">
            {runs.map((run) => (
              <button
                key={run.id}
                type="button"
                onClick={() => setSelectedRun(run.id)}
                className={`flex w-full flex-col gap-1 px-4 py-2.5 text-left text-[12px] hover:bg-arena-bg-subtle ${
                  run.id === selectedRun ? "bg-arena-bg-subtle font-medium" : ""
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] text-arena-text-tertiary">{run.id.slice(-8)}</span>
                  <Badge variant="outline" className="font-mono text-[10px] uppercase">
                    {run.status}
                  </Badge>
                </div>
                <div className="text-arena-text-tertiary">{formatDateTime(run.created_at)}</div>
                {run.summary && typeof run.summary.total === "number" ? (
                  <div className="text-arena-text-tertiary">
                    {run.summary.completed ?? 0}/{run.summary.total} 完成 · judged {run.summary.judged ?? 0}
                  </div>
                ) : null}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-[13px]">运行详情</CardTitle>
        </CardHeader>
        <CardContent>
          {loading || !detail ? (
            <div className="py-8 text-center text-[13px] text-arena-text-tertiary">加载中…</div>
          ) : (
            <RunDetailBody detail={detail} />
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function RunDetailBody({ detail }: { detail: DatasetRunDetail }) {
  const w = detail.summary?.winners
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-[12px] text-arena-text-tertiary">
        <span className="font-mono">{detail.id}</span>
        <span>·</span>
        <Badge variant="outline" className="font-mono uppercase">
          {detail.status}
        </Badge>
        <span>·</span>
        <span>开始 {formatDateTime(detail.created_at)}</span>
        {detail.finished_at ? <span>· 结束 {formatDateTime(detail.finished_at)}</span> : null}
      </div>

      {detail.summary?.total !== undefined ? (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Stat label="总用例" value={String(detail.summary.total ?? 0)} />
          <Stat label="成功 / 失败" value={`${detail.summary.completed ?? 0} / ${detail.summary.failed ?? 0}`} />
          <Stat label="已评判" value={String(detail.summary.judged ?? 0)} />
          <Stat
            label="Winners A/B/T"
            value={w ? `${w.a}/${w.b}/${w.tie}` : "-"}
          />
        </div>
      ) : null}

      <div className="space-y-2">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-arena-text-tertiary">
          逐条结果
        </div>
        <div className="overflow-hidden rounded-arena border border-arena-border">
          <table className="w-full text-left text-[12px]">
            <thead className="bg-arena-bg-subtle text-[11px] uppercase tracking-wider text-arena-text-tertiary">
              <tr>
                <th className="px-3 py-2">#</th>
                <th className="px-3 py-2">Query</th>
                <th className="px-3 py-2">Winner</th>
                <th className="px-3 py-2">状态</th>
              </tr>
            </thead>
            <tbody>
              {detail.items.map((item) => {
                const judge = item.body?.judge
                const winner = judge?.winner || "-"
                return (
                  <tr key={item.id} className="border-t border-arena-border align-top">
                    <td className="px-3 py-2 font-mono text-[11px]">{item.idx + 1}</td>
                    <td className="px-3 py-2">
                      <div className="line-clamp-2 whitespace-pre-wrap">
                        {item.body?.case?.query || "(无 query)"}
                      </div>
                      {item.body?.error ? (
                        <div className="mt-1 text-[11px] text-red-600">{item.body.error}</div>
                      ) : null}
                    </td>
                    <td className="px-3 py-2">
                      <Badge variant="outline" className="font-mono uppercase">
                        {winner}
                      </Badge>
                    </td>
                    <td className="px-3 py-2">
                      <Badge variant="outline" className="font-mono uppercase">
                        {item.status}
                      </Badge>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-arena border border-arena-border bg-arena-bg-subtle px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-arena-text-tertiary">{label}</div>
      <div className="mt-0.5 font-mono text-[14px] font-semibold text-arena-text-primary">{value}</div>
    </div>
  )
}
