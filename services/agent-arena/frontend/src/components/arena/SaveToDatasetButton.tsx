import { FormEvent, useEffect, useState } from "react"
import { BookmarkPlus, Plus } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { apiFetch, readApiJson } from "@/lib/api"
import { normalizeError } from "@/lib/frontendLogger"
import type { Dataset } from "@/lib/types"

export type SaveToDatasetSource =
  | { kind: "arena"; runId: string }
  | { kind: "batch-item"; batchId: string; idx: number; agentKey: string }

export function SaveToDatasetButton({
  source,
  defaultQuery,
  defaultSystem,
  defaultExpectedAnswer = "",
  variant = "outline",
  label = "加入评测集",
  size = "sm",
  disabled,
}: {
  source: SaveToDatasetSource
  defaultQuery?: string
  defaultSystem?: string
  defaultExpectedAnswer?: string
  variant?: "default" | "outline" | "ghost"
  label?: string
  size?: "default" | "sm" | "lg" | "icon"
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState("")
  const [expectedAnswer, setExpectedAnswer] = useState(defaultExpectedAnswer)
  const [tags, setTags] = useState("")
  const [newDatasetName, setNewDatasetName] = useState("")
  const [creatingNew, setCreatingNew] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState("")
  const [savedMessage, setSavedMessage] = useState("")

  async function loadDatasets() {
    setLoading(true)
    setError("")
    try {
      const response = await apiFetch("/datasets")
      const data = await readApiJson<{ datasets: Dataset[] }>(response)
      setDatasets(data.datasets)
      if (!selected && data.datasets[0]) {
        setSelected(data.datasets[0].id)
      }
    } catch (err) {
      setError(normalizeError(err).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (open) {
      setSavedMessage("")
      setExpectedAnswer(defaultExpectedAnswer)
      void loadDatasets()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  async function createDataset() {
    if (!newDatasetName.trim()) return
    setSubmitting(true)
    setError("")
    try {
      const response = await apiFetch("/datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newDatasetName, description: "" }),
      })
      const ds = await readApiJson<Dataset>(response)
      setNewDatasetName("")
      setCreatingNew(false)
      await loadDatasets()
      setSelected(ds.id)
    } catch (err) {
      setError(normalizeError(err).message)
    } finally {
      setSubmitting(false)
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!selected) {
      setError("请选择目标评测集。")
      return
    }
    setSubmitting(true)
    setError("")
    try {
      const tagList = tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean)
      const basePayload = {
        dataset_id: selected,
        expected_answer: expectedAnswer,
        tags: tagList,
        query_override: defaultQuery ?? null,
        system_override: defaultSystem ?? null,
      }
      const endpoint =
        source.kind === "arena"
          ? `/arena-runs/${source.runId}/save-to-dataset`
          : `/batch-items/save-to-dataset`
      const sourceFields =
        source.kind === "arena"
          ? {}
          : {
              batch_id: source.batchId,
              idx: source.idx,
              agent_key: source.agentKey,
            }
      const response = await apiFetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...basePayload, ...sourceFields }),
      })
      await readApiJson(response)
      setSavedMessage("已保存到评测集。")
      setTags("")
    } catch (err) {
      setError(normalizeError(err).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <>
      <Button
        type="button"
        variant={variant}
        size={size}
        disabled={disabled}
        onClick={() => setOpen(true)}
      >
        <BookmarkPlus className="size-3.5" />
        {label}
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="w-[min(620px,96vw)]">
          <form onSubmit={submit} className="flex flex-col gap-4 p-6">
            <div>
              <DialogTitle>加入评测集</DialogTitle>
              <DialogDescription>
                把当前 Arena 运行的输入存为评测用例。Query / System 来自本次运行；可补一份黄金答案后供 Judge 评分。
              </DialogDescription>
            </div>

            <div className="space-y-1.5">
              <Label className="text-[12px]">目标评测集</Label>
              {loading ? (
                <div className="text-[12px] text-arena-text-tertiary">加载中…</div>
              ) : datasets.length === 0 && !creatingNew ? (
                <div className="rounded-arena border border-dashed border-arena-border bg-arena-bg-subtle px-3 py-3 text-[12px] text-arena-text-tertiary">
                  尚无评测集。
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="ml-2 inline-flex"
                    onClick={() => setCreatingNew(true)}
                  >
                    <Plus className="size-3.5" />
                    新建一个
                  </Button>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <select
                    value={selected}
                    onChange={(e) => setSelected(e.target.value)}
                    className="h-9 flex-1 rounded-arena border border-arena-border bg-white px-3 text-[13px]"
                  >
                    {datasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name} ({d.case_count ?? 0})
                      </option>
                    ))}
                  </select>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setCreatingNew((v) => !v)}
                  >
                    <Plus className="size-3.5" />
                    新建
                  </Button>
                </div>
              )}
            </div>

            {creatingNew ? (
              <div className="flex items-end gap-2 rounded-arena border border-arena-border bg-arena-bg-subtle px-3 py-2">
                <div className="flex-1 space-y-1.5">
                  <Label className="text-[12px]">新评测集名称</Label>
                  <Input
                    value={newDatasetName}
                    onChange={(e) => setNewDatasetName(e.target.value)}
                    placeholder="比如 金融 FAQ v1"
                  />
                </div>
                <Button type="button" size="sm" disabled={submitting} onClick={createDataset}>
                  创建
                </Button>
              </div>
            ) : null}

            <div className="space-y-1.5">
              <Label className="text-[12px]">黄金答案（可选）</Label>
              <Textarea
                rows={3}
                value={expectedAnswer}
                onChange={(e) => setExpectedAnswer(e.target.value)}
                placeholder="为这条用例补一个参考答案；留空则只评过程"
              />
            </div>

            <div className="space-y-1.5">
              <Label className="text-[12px]">Tags（逗号分隔）</Label>
              <Input
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                placeholder="regression, hard, multi-turn"
              />
            </div>

            {error ? <div className="text-[12px] text-red-600">{error}</div> : null}
            {savedMessage ? (
              <div className="text-[12px] text-green-700">{savedMessage}</div>
            ) : null}

            <div className="flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={() => setOpen(false)}>
                关闭
              </Button>
              <Button type="submit" disabled={submitting || !selected}>
                {submitting ? "保存中…" : "保存为用例"}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </>
  )
}
