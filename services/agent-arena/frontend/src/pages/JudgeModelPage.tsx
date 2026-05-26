import { FormEvent, useEffect, useState } from "react"
import { AlertTriangle, RefreshCcw, Save, ShieldCheck } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { apiFetch, isUnauthorizedError, readApiJson } from "@/lib/api"
import { normalizeError } from "@/lib/frontendLogger"
import type { JudgeConfigDef } from "@/lib/types"

type FormState = {
  base_url: string
  model: string
  api_key_env: string
}

const EMPTY_FORM: FormState = { base_url: "", model: "", api_key_env: "" }

export function JudgeModelPage({ onUnauthorized }: { onUnauthorized?: () => void }) {
  const [config, setConfig] = useState<JudgeConfigDef | null>(null)
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState("")
  const [formError, setFormError] = useState("")
  const [savedAt, setSavedAt] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    setError("")
    try {
      const response = await apiFetch("/judge/config")
      const data = await readApiJson<JudgeConfigDef>(response)
      setConfig(data)
      setForm({
        base_url: data.base_url,
        model: data.model,
        api_key_env: data.api_key_env.name,
      })
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

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!form.base_url.trim() || !form.model.trim()) {
      setFormError("Base URL 和模型不能为空。")
      return
    }
    setSaving(true)
    setFormError("")
    try {
      const response = await apiFetch("/judge/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          base_url: form.base_url,
          model: form.model,
          api_key_env: form.api_key_env,
        }),
      })
      const next = await readApiJson<JudgeConfigDef>(response)
      setConfig(next)
      setForm({
        base_url: next.base_url,
        model: next.model,
        api_key_env: next.api_key_env.name,
      })
      setSavedAt(new Date().toLocaleTimeString())
    } catch (err) {
      if (isUnauthorizedError(err)) {
        onUnauthorized?.()
        return
      }
      setFormError(normalizeError(err).message)
    } finally {
      setSaving(false)
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
              <ShieldCheck className="size-4 text-arena-accent" />
              Judge 模型
            </CardTitle>
            <p className="mt-1 text-[12px] text-arena-text-tertiary">
              竞技场和评测集都会调用这个模型对 A / B 输出打分。API Key 通过环境变量名引用，密钥本身不进数据库。
            </p>
          </div>
          <div className="flex items-center gap-2">
            {config ? (
              <Badge variant="outline" className="font-mono uppercase">
                source: {config.source}
              </Badge>
            ) : null}
            <Button type="button" variant="ghost" size="sm" onClick={load}>
              <RefreshCcw className="size-3.5" />
              刷新
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="py-8 text-center text-[13px] text-arena-text-tertiary">加载中…</div>
          ) : (
            <form onSubmit={submit} className="grid gap-4 md:grid-cols-2">
              <div className="space-y-1.5 md:col-span-2">
                <Label className="text-[12px]">Base URL</Label>
                <Input
                  value={form.base_url}
                  onChange={(e) => setForm({ ...form, base_url: e.target.value })}
                  placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1"
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-[12px]">模型</Label>
                <Input
                  value={form.model}
                  onChange={(e) => setForm({ ...form, model: e.target.value })}
                  placeholder="qwen-max"
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-[12px]">API Key 环境变量名</Label>
                <Input
                  value={form.api_key_env}
                  onChange={(e) => setForm({ ...form, api_key_env: e.target.value })}
                  placeholder="JUDGE_API_KEY"
                />
              </div>

              {config ? (
                <div className="md:col-span-2 rounded-arena border border-arena-border bg-arena-bg-subtle px-3 py-2 text-[11px]">
                  <div className="mb-1 font-semibold uppercase tracking-wider text-arena-text-tertiary">
                    当前环境变量状态
                  </div>
                  {config.api_key_env.name ? (
                    <div className="flex items-center gap-2 font-mono">
                      <span className="rounded-arena-sm bg-white px-1.5 py-0.5">
                        ${config.api_key_env.name}
                      </span>
                      {config.api_key_env.present ? (
                        <Badge variant="outline" className="text-green-700">
                          {config.api_key_env.preview || "set"}
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="text-amber-700">
                          env 缺失
                        </Badge>
                      )}
                    </div>
                  ) : (
                    <div className="text-arena-text-tertiary">未配置环境变量名。</div>
                  )}
                </div>
              ) : null}

              {formError ? (
                <div className="md:col-span-2 text-[12px] text-red-600">{formError}</div>
              ) : null}

              <div className="md:col-span-2 flex items-center justify-between gap-2 pt-1">
                <div className="text-[11px] text-arena-text-tertiary">
                  {savedAt ? `已保存 · ${savedAt}` : "未保存改动"}
                </div>
                <Button type="submit" disabled={saving}>
                  <Save className="size-3.5" />
                  {saving ? "保存中…" : "保存配置"}
                </Button>
              </div>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
