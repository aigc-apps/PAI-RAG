import { FormEvent } from "react"
import { AlertTriangle, Eraser, Loader2, Play, Scale } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { AgentResultPanel } from "@/components/arena/AgentResultPanel"
import { CodeBlock } from "@/components/arena/CodeBlock"
import { JudgePanel } from "@/components/arena/JudgePanel"
import type {
  CompareResponse,
  ConfigResponse,
  JudgeResponse,
} from "@/lib/types"

export function ArenaPage({
  input,
  setInput,
  system,
  setSystem,
  temperature,
  setTemperature,
  maxTokens,
  setMaxTokens,
  parsedTemperature,
  parsedMaxTokens,
  loading,
  judgeLoading,
  result,
  judge,
  error,
  canJudge,
  config,
  onSubmit,
  onJudge,
  onClear,
}: {
  input: string
  setInput: (value: string) => void
  system: string
  setSystem: (value: string) => void
  temperature: string
  setTemperature: (value: string) => void
  maxTokens: string
  setMaxTokens: (value: string) => void
  parsedTemperature: number
  parsedMaxTokens: number | undefined
  loading: boolean
  judgeLoading: boolean
  result: CompareResponse | null
  judge: JudgeResponse | null
  error: string
  canJudge: boolean
  config: ConfigResponse | null
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  onJudge: () => void
  onClear: () => void
}) {
  return (
    <div className="grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
      <Card className="h-fit">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>请求配置</CardTitle>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onClear}
            disabled={loading || judgeLoading}
          >
            <Eraser className="size-3.5" />
            清空
          </Button>
        </CardHeader>
        <CardContent className="p-[18px]">
          <form className="space-y-3.5" onSubmit={onSubmit}>
            <div className="space-y-1.5">
              <Label htmlFor="prompt" className="flex items-center gap-1.5 text-[12px] font-semibold uppercase tracking-wider text-arena-text-secondary">
                输入 Query
                <span className="font-mono text-[10px] font-normal text-arena-text-tertiary">user prompt</span>
              </Label>
              <Textarea
                id="prompt"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="输入要同时发送给两个 Agent 的问题"
                className="min-h-[140px] resize-y"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="system" className="flex items-center gap-1.5 text-[12px] font-semibold uppercase tracking-wider text-arena-text-secondary">
                System Prompt
                <span className="font-mono text-[10px] font-normal text-arena-text-tertiary">可选</span>
              </Label>
              <Textarea
                id="system"
                value={system}
                onChange={(event) => setSystem(event.target.value)}
                placeholder="可选"
                className="min-h-[80px] resize-y"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label htmlFor="temperature" className="text-[12px] font-semibold uppercase tracking-wider text-arena-text-secondary">
                  Temperature
                </Label>
                <Input
                  id="temperature"
                  value={temperature}
                  onChange={(event) => setTemperature(event.target.value)}
                  inputMode="decimal"
                  className="font-mono"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="maxTokens" className="text-[12px] font-semibold uppercase tracking-wider text-arena-text-secondary">
                  Max tokens
                </Label>
                <Input
                  id="maxTokens"
                  value={maxTokens}
                  onChange={(event) => setMaxTokens(event.target.value)}
                  inputMode="numeric"
                  className="font-mono"
                />
              </div>
            </div>

            {error ? (
              <Alert variant="destructive">
                <AlertTriangle className="size-4" />
                <AlertTitle>提交失败</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            ) : null}

            <div className="flex flex-col gap-2 pt-2">
              <Button type="submit" disabled={loading || judgeLoading} className="w-full">
                {loading ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
                发起对比
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={onJudge}
                disabled={!canJudge || judgeLoading}
                className="w-full"
              >
                {judgeLoading ? <Loader2 className="size-4 animate-spin" /> : <Scale className="size-4" />}
                运行 Judge
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <div className="space-y-5">
        <div className="grid gap-5 xl:grid-cols-2">
          <AgentResultPanel
            id="a"
            title="Agent A"
            config={config?.agents.a}
            result={result?.agents.a}
            loading={loading}
            input={input}
          />
          <AgentResultPanel
            id="b"
            title="Agent B"
            config={config?.agents.b}
            result={result?.agents.b}
            loading={loading}
            input={input}
          />
        </div>

        <JudgePanel
          judge={judge}
          judgeLoading={judgeLoading}
          canJudge={canJudge}
          config={config}
          onJudge={onJudge}
        />

        <Card>
          <CardHeader>
            <CardTitle>Debug 信息</CardTitle>
          </CardHeader>
          <CardContent className="p-[18px]">
            <Tabs defaultValue="request">
              <TabsList>
                <TabsTrigger value="request">Request</TabsTrigger>
                <TabsTrigger value="response">Compare response</TabsTrigger>
                <TabsTrigger value="judge">Judge JSON</TabsTrigger>
                <TabsTrigger value="config">Config snapshot</TabsTrigger>
              </TabsList>
              <TabsContent value="request" className="mt-3">
                <CodeBlock
                  value={JSON.stringify(
                    { input, system, temperature: parsedTemperature, max_tokens: parsedMaxTokens },
                    null,
                    2,
                  )}
                />
              </TabsContent>
              <TabsContent value="response" className="mt-3">
                <CodeBlock value={result ? JSON.stringify(result, null, 2) : "尚无对比结果"} />
              </TabsContent>
              <TabsContent value="judge" className="mt-3">
                <CodeBlock value={judge ? JSON.stringify(judge, null, 2) : "尚无 Judge 结果"} />
              </TabsContent>
              <TabsContent value="config" className="mt-3">
                <CodeBlock value={config ? JSON.stringify(config, null, 2) : "配置尚未加载"} />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
