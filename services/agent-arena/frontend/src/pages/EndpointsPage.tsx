import { FormEvent, useEffect, useState } from "react"
import { AlertTriangle, Pencil, Plus, RefreshCcw, Save, Trash2 } from "lucide-react"

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
import { Textarea } from "@/components/ui/textarea"
import { apiFetch, isUnauthorizedError, readApiJson } from "@/lib/api"
import { normalizeError } from "@/lib/frontendLogger"
import type { ActivePair, AgentDef, AgentListResponse } from "@/lib/types"

const TRACE_MODES = ["responses", "chat", "runs", "openclaw"] as const

type FormState = {
  id: string | null
  name: string
  base_url: string
  model: string
  trace_mode: (typeof TRACE_MODES)[number]
  api_key_env: string
  runs_base_url: string
  description: string
}

const EMPTY_FORM: FormState = {
  id: null,
  name: "",
  base_url: "",
  model: "",
  trace_mode: "responses",
  api_key_env: "",
  runs_base_url: "",
  description: "",
}

export function EndpointsPage({ onUnauthorized }: { onUnauthorized?: () => void }) {
  const [agents, setAgents] = useState<AgentDef[]>([])
  const [pair, setPair] = useState<ActivePair>({
    a_agent_id: null,
    b_agent_id: null,
    updated_at: null,
  })
  const [pairDirty, setPairDirty] = useState(false)
  const [pairSaving, setPairSaving] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [formError, setFormError] = useState("")

  async function load() {
    setLoading(true)
    setError("")
    try {
      const response = await apiFetch("/agents")
      const data = await readApiJson<AgentListResponse>(response)
      setAgents(data.agents)
      setPair(data.active_pair)
      setPairDirty(false)
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
    setForm(EMPTY_FORM)
    setFormError("")
    setDialogOpen(true)
  }

  function openEdit(agent: AgentDef) {
    setForm({
      id: agent.id,
      name: agent.name,
      base_url: agent.base_url,
      model: agent.model,
      trace_mode: agent.trace_mode,
      api_key_env: agent.api_key_env.name,
      runs_base_url: agent.runs_base_url,
      description: agent.description,
    })
    setFormError("")
    setDialogOpen(true)
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!form.name.trim() || !form.base_url.trim() || !form.model.trim()) {
      setFormError("名称、Base URL、模型不能为空。")
      return
    }
    setSubmitting(true)
    setFormError("")
    try {
      const path = form.id ? `/agents/${form.id}` : "/agents"
      const method = form.id ? "PATCH" : "POST"
      const response = await apiFetch(path, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: form.name,
          base_url: form.base_url,
          model: form.model,
          trace_mode: form.trace_mode,
          api_key_env: form.api_key_env,
          runs_base_url: form.runs_base_url,
          description: form.description,
        }),
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

  async function remove(agent: AgentDef) {
    if (!window.confirm(`删除 Agent「${agent.name}」？此操作不可撤销。`)) return
    try {
      const response = await apiFetch(`/agents/${agent.id}`, { method: "DELETE" })
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

  async function savePair() {
    setPairSaving(true)
    try {
      const response = await apiFetch("/agents/active-pair", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          a_agent_id: pair.a_agent_id,
          b_agent_id: pair.b_agent_id,
        }),
      })
      await readApiJson(response)
      setPairDirty(false)
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
      setError(normalizeError(err).message)
    } finally {
      setPairSaving(false)
    }
  }

  function selectPair(slot: "a" | "b", value: string) {
    setPair((prev) => ({
      ...prev,
      [slot === "a" ? "a_agent_id" : "b_agent_id"]: value || null,
    }))
    setPairDirty(true)
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
            <CardTitle>当前 A / B 绑定</CardTitle>
            <p className="mt-1 text-[12px] text-arena-text-tertiary">
              竞技场、稳定性测试、评测集都会调用这两个 Agent。可随时切换。
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button type="button" variant="ghost" size="sm" onClick={load}>
              <RefreshCcw className="size-3.5" />
              刷新
            </Button>
            <Button type="button" size="sm" disabled={!pairDirty || pairSaving} onClick={savePair}>
              <Save className="size-3.5" />
              {pairSaving ? "保存中…" : "保存绑定"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <PairPicker slot="A" value={pair.a_agent_id} agents={agents} onChange={(v) => selectPair("a", v)} />
          <PairPicker slot="B" value={pair.b_agent_id} agents={agents} onChange={(v) => selectPair("b", v)} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0">
          <div>
            <CardTitle>Agent 库</CardTitle>
            <p className="mt-1 text-[12px] text-arena-text-tertiary">
              API Key 不保存在数据库；这里只记录环境变量名，调用时从环境读取。
            </p>
          </div>
          <Button type="button" size="sm" onClick={openCreate}>
            <Plus className="size-3.5" />
            新增 Agent
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="px-6 py-10 text-center text-[13px] text-arena-text-tertiary">加载中…</div>
          ) : agents.length === 0 ? (
            <div className="px-6 py-10 text-center text-[13px] text-arena-text-tertiary">
              暂无 Agent。点击右上「新增 Agent」开始配置。
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-[13px]">
                <thead className="border-b border-arena-border bg-arena-bg-subtle">
                  <tr className="text-[11px] font-semibold uppercase tracking-wider text-arena-text-tertiary">
                    <th className="px-4 py-2.5">名称</th>
                    <th className="px-4 py-2.5">Base URL</th>
                    <th className="px-4 py-2.5">模型</th>
                    <th className="px-4 py-2.5">Trace mode</th>
                    <th className="px-4 py-2.5">API Key (env)</th>
                    <th className="px-4 py-2.5 text-right">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {agents.map((agent) => (
                    <tr key={agent.id} className="border-b border-arena-border last:border-0 hover:bg-arena-bg-subtle">
                      <td className="px-4 py-3 align-top">
                        <div className="font-medium text-arena-text-primary">{agent.name}</div>
                        {agent.description ? (
                          <div className="mt-0.5 text-[11px] text-arena-text-tertiary">{agent.description}</div>
                        ) : null}
                      </td>
                      <td className="px-4 py-3 align-top font-mono text-[11px] text-arena-text-secondary">
                        {agent.base_url}
                      </td>
                      <td className="px-4 py-3 align-top font-mono text-[11px] text-arena-text-secondary">
                        {agent.model}
                      </td>
                      <td className="px-4 py-3 align-top">
                        <Badge variant="outline" className="font-mono uppercase">
                          {agent.trace_mode}
                        </Badge>
                      </td>
                      <td className="px-4 py-3 align-top text-[12px]">
                        <ApiKeyCell status={agent.api_key_env} />
                      </td>
                      <td className="px-4 py-3 text-right align-top">
                        <div className="inline-flex gap-1">
                          <Button type="button" variant="ghost" size="sm" onClick={() => openEdit(agent)}>
                            <Pencil className="size-3.5" />
                          </Button>
                          <Button type="button" variant="ghost" size="sm" onClick={() => remove(agent)}>
                            <Trash2 className="size-3.5 text-red-600" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="w-[min(680px,96vw)]">
          <form onSubmit={submit} className="flex flex-col gap-4 p-6">
            <div>
              <DialogTitle>{form.id ? "编辑 Agent" : "新增 Agent"}</DialogTitle>
              <DialogDescription>
                Trace mode 决定后端调用哪种 API：responses 用 /v1/responses；chat 用 /v1/chat/completions；runs 用 /v1/runs；openclaw 用 OpenClaw cookie 会话和 /api/chat。
              </DialogDescription>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <Field label="名称">
                <Input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="比如 pairag-agent-v2" />
              </Field>
              <Field label="模型">
                <Input value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} placeholder="hermes-agent" />
              </Field>
              <Field label="Base URL" className="md:col-span-2">
                <Input value={form.base_url} onChange={(e) => setForm({ ...form, base_url: e.target.value })} placeholder="https://host/v1" />
              </Field>
              <Field label="Trace mode">
                <select
                  value={form.trace_mode}
                  onChange={(e) => setForm({ ...form, trace_mode: e.target.value as FormState["trace_mode"] })}
                  className="h-9 w-full rounded-arena border border-arena-border bg-white px-3 text-[13px]"
                >
                  {TRACE_MODES.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </Field>
              <Field label="API Key 环境变量名">
                <Input value={form.api_key_env} onChange={(e) => setForm({ ...form, api_key_env: e.target.value })} placeholder="AGENT_A_API_KEY" />
              </Field>
              <Field label="Runs base URL（仅 runs 模式）" className="md:col-span-2">
                <Input value={form.runs_base_url} onChange={(e) => setForm({ ...form, runs_base_url: e.target.value })} placeholder="可选；为空时复用 Base URL" />
              </Field>
              <Field label="说明" className="md:col-span-2">
                <Textarea rows={2} value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
              </Field>
            </div>
            {formError ? (
              <div className="text-[12px] text-red-600">{formError}</div>
            ) : null}
            <div className="flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={() => setDialogOpen(false)}>取消</Button>
              <Button type="submit" disabled={submitting}>
                {submitting ? "保存中…" : "保存"}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  )
}

function PairPicker({
  slot,
  value,
  agents,
  onChange,
}: {
  slot: "A" | "B"
  value: string | null
  agents: AgentDef[]
  onChange: (value: string) => void
}) {
  const selected = agents.find((a) => a.id === value) || null
  return (
    <div className="space-y-2">
      <Label className="flex items-center gap-2 text-[12px]">
        <span className="inline-grid size-5 place-items-center rounded-full bg-arena-accent-soft font-mono text-[11px] font-bold text-arena-accent-press">
          {slot}
        </span>
        Agent {slot}
      </Label>
      <select
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        className="h-10 w-full rounded-arena border border-arena-border bg-white px-3 text-[13px]"
      >
        <option value="">（未绑定，调用时回退环境变量）</option>
        {agents.map((agent) => (
          <option key={agent.id} value={agent.id}>{agent.name}</option>
        ))}
      </select>
      {selected ? (
        <div className="rounded-arena border border-arena-border bg-arena-bg-subtle px-3 py-2 text-[11px]">
          <div className="font-mono text-arena-text-secondary">{selected.base_url}</div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-arena-text-tertiary">
            <Badge variant="outline" className="font-mono uppercase">{selected.trace_mode}</Badge>
            <span className="font-mono">{selected.model}</span>
            <ApiKeyCell status={selected.api_key_env} inline />
          </div>
        </div>
      ) : null}
    </div>
  )
}

function ApiKeyCell({
  status,
  inline,
}: {
  status: AgentDef["api_key_env"]
  inline?: boolean
}) {
  if (!status.name) {
    return <span className="text-arena-text-tertiary">{inline ? "no key" : "未配置"}</span>
  }
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[11px]">
      <span className="rounded-arena-sm bg-arena-bg-subtle px-1.5 py-0.5">${status.name}</span>
      {status.present ? (
        <Badge variant="outline" className="text-green-700">{status.preview || "set"}</Badge>
      ) : (
        <Badge variant="outline" className="text-amber-700">env 缺失</Badge>
      )}
    </span>
  )
}

function Field({
  label,
  className,
  children,
}: {
  label: string
  className?: string
  children: React.ReactNode
}) {
  return (
    <div className={`space-y-1.5 ${className || ""}`}>
      <Label className="text-[12px]">{label}</Label>
      {children}
    </div>
  )
}
