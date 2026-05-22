import { FormEvent } from "react"
import { AlertTriangle, Eraser, Gavel, Loader2, SendHorizontal } from "lucide-react"

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
import { Separator } from "@/components/ui/separator"
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
    <section className="grid gap-5 xl:grid-cols-[420px_1fr]">
      <Card className="h-fit">
        <CardHeader>
          <CardTitle>输入</CardTitle>
          <CardDescription>
            请求由后端转发，浏览器不会接触 Agent 或 Judge API key。
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={onSubmit}>
            <div className="space-y-2">
              <Label htmlFor="prompt">User prompt</Label>
              <Textarea
                id="prompt"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="输入要同时发送给两个 Agent 的问题"
                className="min-h-[180px] resize-y"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="system">System prompt</Label>
              <Textarea
                id="system"
                value={system}
                onChange={(event) => setSystem(event.target.value)}
                placeholder="可选"
                className="min-h-[96px] resize-y"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <Label htmlFor="temperature">Temperature</Label>
                <Input
                  id="temperature"
                  value={temperature}
                  onChange={(event) => setTemperature(event.target.value)}
                  inputMode="decimal"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="maxTokens">Max tokens</Label>
                <Input
                  id="maxTokens"
                  value={maxTokens}
                  onChange={(event) => setMaxTokens(event.target.value)}
                  inputMode="numeric"
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
            <div className="flex flex-wrap gap-2">
              <Button type="submit" disabled={loading || judgeLoading}>
                {loading ? <Loader2 className="size-4 animate-spin" /> : <SendHorizontal className="size-4" />}
                开始对比
              </Button>
              <Button type="button" variant="secondary" onClick={onJudge} disabled={!canJudge || judgeLoading}>
                {judgeLoading ? <Loader2 className="size-4 animate-spin" /> : <Gavel className="size-4" />}
                对比
              </Button>
              <Button type="button" variant="outline" onClick={onClear} disabled={loading || judgeLoading}>
                <Eraser className="size-4" />
                清空
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <div className="space-y-5">
        <div className="grid gap-5 lg:grid-cols-2">
          <AgentResultPanel title="Agent A" config={config?.agents.a} result={result?.agents.a} loading={loading} />
          <AgentResultPanel title="Agent B" config={config?.agents.b} result={result?.agents.b} loading={loading} />
        </div>

        <JudgePanel judge={judge} judgeLoading={judgeLoading} canJudge={canJudge} config={config} onJudge={onJudge} />

        <Card>
          <CardHeader className="pb-3">
            <CardTitle>调试信息</CardTitle>
            <CardDescription>查看实际请求 payload、Agent 返回和 Judge 返回。</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="request">
              <TabsList>
                <TabsTrigger value="request">Request</TabsTrigger>
                <TabsTrigger value="response">Response</TabsTrigger>
                <TabsTrigger value="judge">Judge</TabsTrigger>
                <TabsTrigger value="config">Config</TabsTrigger>
              </TabsList>
              <Separator className="my-3" />
              <TabsContent value="request">
                <CodeBlock
                  value={JSON.stringify(
                    { input, system, temperature: parsedTemperature, max_tokens: parsedMaxTokens },
                    null,
                    2,
                  )}
                />
              </TabsContent>
              <TabsContent value="response">
                <CodeBlock value={result ? JSON.stringify(result, null, 2) : "尚无对比结果"} />
              </TabsContent>
              <TabsContent value="judge">
                <CodeBlock value={judge ? JSON.stringify(judge, null, 2) : "尚无 Judge 结果"} />
              </TabsContent>
              <TabsContent value="config">
                <CodeBlock value={config ? JSON.stringify(config, null, 2) : "配置尚未加载"} />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </section>
  )
}
