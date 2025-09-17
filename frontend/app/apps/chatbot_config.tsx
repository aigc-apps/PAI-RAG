'use client';
import React, { useState, useEffect, FC } from 'react';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { ChevronDownIcon, Terminal } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

import { Button } from '@/components/ui/button';
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { useRouter } from 'next/navigation';
import {PLAN_PROMPT, ACT_PROMPT, SUMMARY_PROMPT} from '../common/prompts';

// Add import for ResettableTextarea
import { ResettableTextarea } from '@/app/apps/resetable_textarea';
import { toast } from 'sonner';


interface PromptConfig {
  plan: string;
  act: string;
  summary: string;
};


export interface Chatbot {
  id: string;
  app_id: string;
  description: string;
  enable_search: boolean;
  enable_agent: boolean;
  mcp_ids: string[];
  kb_ids: string[];
  model_id: string;
  updated_at: string;
  enable_input_guardrail: boolean;
  enable_output_guardrail: boolean;
  guardrail_hint: string;
  prompts: PromptConfig;
}


interface ChatbotConfigProps {
  chatbotId: string | undefined;
}


const default_chat_config = {
  id: '',
  app_id: '',
  description: '',
  enable_search: false,
  mcp_ids: [],
  kb_ids: [],
  model_id: "",
  updated_at: "",
  enable_agent: false,
  enable_input_guardrail: false,
  enable_output_guardrail: false,
  guardrail_hint: "作为人工智能助手，我无法回应包含不当或敏感信息的内容。",
  prompts: {
    plan: PLAN_PROMPT,
    act: ACT_PROMPT,
    summary: SUMMARY_PROMPT,
  }
};

// 知识库配置卡片
export const ChatbotConfigCard: FC<ChatbotConfigProps> = ({
  chatbotId,
}) => {
  const [botConfig, setBotConfig] = useState<Chatbot>(default_chat_config);
  const [llms, setLlms] = useState<LlmConfig[]>([]);
  const [mcps, setMcps] = useState<McpConfig[]>([]);
  const [kbs, setKbs] = useState<KbConfig[]>([]);
  const [selectedKbNames, setSelectedKbNames] = useState<string[]>([]);
  const [selectedMcpNames, setSelectedMcpNames] = useState<string[]>([]);
  const [saveErrorMsg, setSaveErrorMsg] = useState('');
  const isCreate: boolean = chatbotId === undefined || chatbotId === '';
  const [planPrompt, setPlanPrompt] = useState('');
  const [actPrompt, setActPrompt] = useState('');
  const [summarizePrompt, setSummarizePrompt] = useState('');

  const router = useRouter();
  // const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        if (!isCreate)
        {
          const [llmRes, mcpRes, kbRes, botRes] = await Promise.all([
            fetch(`/api/config/llms`),
            fetch(`/api/config/mcps`),
            fetch(`/api/config/knowledgebases`),
            fetch(`/api/config/apps?app_id=${chatbotId}`)]);
          const llmData = (await llmRes.json())?.data.items || [];
          console.log('llmData', llmData);
          setLlms([...llmData]);

          const [] = await Promise.all([]);

          const mcpData =
            ((await mcpRes.json())?.data.items as McpConfig[]) || [];
          console.log('mcpData', mcpData);
          setMcps([...mcpData]);

          const kbData = ((await kbRes.json())?.data.items as KbConfig[]) || [];
          console.log('kbData', kbData);
          setKbs([...kbData]);
          const botData = await botRes.json();
          botData.data.kb_ids = botData.data.kb_ids.filter(
            (kb_id: string) => {
              return kbData.some((kb: any) => kb.id === kb_id);
            }
          )
          botData.data.mcp_ids = botData.data.mcp_ids.filter(
            (mcp_id: string) => {
              return mcpData.some((mcp: any) => mcp.id === mcp_id);
            }
          )

          setBotConfig(botData.data);
          console.log('chatbotData: ', botData.data);

          const kbnames = kbData
            .filter((item) => botData.data.kb_ids.includes(item.id))
            .map((item) => item.name);
          setSelectedKbNames([...kbnames]);
          console.log('selectedKbNames', kbnames);

          const mcpnames = mcpData
            .filter((item) => botData.data.mcp_ids.includes(item.id))
            .map((item) => item.name);
          setSelectedMcpNames([...mcpnames]);
          console.log('selectedMcpNames', mcpnames);

          setPlanPrompt(botData.data.prompts?.plan || PLAN_PROMPT);
          setActPrompt(botData.data.prompts?.act || ACT_PROMPT);
          setSummarizePrompt(botData.data.prompts?.summary || SUMMARY_PROMPT);
        }
        else
        {
          const [llmRes, mcpRes, kbRes] = await Promise.all([
            fetch(`/api/config/llms`),
            fetch(`/api/config/mcps`),
            fetch(`/api/config/knowledgebases`)]);

          const llmData = (await llmRes.json())?.data.items || [];
          console.log('llmData', llmData);
          setLlms([...llmData]);

          const [] = await Promise.all([]);

          const mcpData =
            ((await mcpRes.json())?.data.items as McpConfig[]) || [];
          console.log('mcpData', mcpData);
          setMcps([...mcpData]);

          const kbData = ((await kbRes.json())?.data.items as KbConfig[]) || [];
          console.log('kbData', kbData);
          setKbs([...kbData]);

          setPlanPrompt(PLAN_PROMPT);
          setActPrompt(ACT_PROMPT);
          setSummarizePrompt(SUMMARY_PROMPT);
        }
      } catch (err: unknown) {
        console.log(err || '加载失败');
      }
    };
    fetchModelConfigs();
  }, [chatbotId, isCreate]);

  const handleSaveChatConfig = async () => {
    console.log('保存应用结果:', botConfig);
    const submit_url = isCreate
      ? `/api/config/apps`
      : `/api/config/apps/${botConfig.id}`;
    const updateMethod = isCreate ? 'POST' : 'PUT';
    try {
      const res = await fetch(submit_url, {
        method: updateMethod,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(botConfig), // 包装为数组
      });

      if (!res.ok) throw new Error(`保存应用失败: ${await res.text()}`);
      router.push('/apps');
      setSaveErrorMsg('');
      // onSaveSuccess(jsondata.data as KbConfig);
    } catch (err: any) {
      console.log('保存应用失败', err.message);
      setSaveErrorMsg(err.message);
    }
  };

  const handleKbSelect = (kb_id: string, kb_name: string, checked: boolean) => {
    console.log('handleKbSelect', kb_id, kb_name, checked);
    if (checked) {
      const kb_ids = botConfig.kb_ids.includes(kb_id)
        ? botConfig.kb_ids
        : [...botConfig.kb_ids, kb_id];
      setBotConfig((prev) => ({ ...prev, kb_ids: kb_ids }));
      if (!selectedKbNames.includes(kb_name)) {
        setSelectedKbNames((prev) => [...prev, kb_name]);
      }
    } else {
      const kb_ids = botConfig.kb_ids.filter((id) => id !== kb_id);
      setBotConfig((prev) => ({ ...prev, kb_ids: kb_ids }));
      if (selectedKbNames.includes(kb_name)) {
        setSelectedKbNames((prev) => prev.filter((name) => name !== kb_name));
      }
    }
  };
  const handleMcpSelect = (
    mcp_id: string,
    mcp_name: string,
    checked: boolean,
  ) => {
    if (checked) {
      const mcp_ids = botConfig.mcp_ids.includes(mcp_id)
        ? botConfig.mcp_ids
        : [...botConfig.mcp_ids, mcp_id];
      setBotConfig((prev) => ({ ...prev, mcp_ids: mcp_ids }));
      if (!selectedMcpNames.includes(mcp_name)) {
        setSelectedMcpNames((prev) => [...prev, mcp_name]);
      }
    } else {
      const mcp_ids = botConfig.mcp_ids.filter((id) => id !== mcp_id);
      setBotConfig((prev) => ({ ...prev, mcp_ids: mcp_ids }));
      if (selectedMcpNames.includes(mcp_name)) {
        setSelectedMcpNames((prev) => prev.filter((name) => name !== mcp_name));
      }
    }
  };

  return (
    <div className="grid gap-4 py-6 px-6">
      <div className="text-xl font-bold">
        {isCreate ? '新建应用' : '编辑应用'}
      </div>
      <div className="space-y-2">
        <Label htmlFor="app-id">
          App ID <span className="text-destructive">*</span>
        </Label>
        <Input
          id="appid"
          value={botConfig.app_id}
          onChange={(e) =>
            setBotConfig((prev) => ({ ...prev, app_id: e.target.value }))
          }
          placeholder="请输入应用ID, 如chatbot"
          required
        />
        <p className="text-sm text-muted-foreground">
          可输入大小写字母和数字,必须字母开头,3-64个字符。
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="description">描述</Label>
        <Textarea
          id="description"
          value={botConfig.description}
          onChange={(e) =>
            setBotConfig((prev) => ({
              ...prev,
              description: e.target.value,
            }))
          }
          placeholder="应用描述（可选）"
          rows={3}
        />
      </div>
      <div className="flex">
        <Label htmlFor="basemodel" className="w-[90px]">
          基模型选择 <span className="text-destructive">*</span>{' '}
        </Label>
        <div className="px-6">
          {llms.length > 0 ? (
            <Select
              value={botConfig.model_id}
              onValueChange={(value) =>
                setBotConfig((prev) => ({
                  ...prev,
                  model_id: value,
                }))
              }
            >
              <SelectTrigger>
                <SelectValue placeholder="请选择基模型" />
              </SelectTrigger>
              <SelectContent>
                {llms.map((llm) => (
                  <SelectItem key={llm.id} value={llm.model_id}>
                    {llm.model_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <div>
              <p className="text-sm text-muted-foreground">尚未配置大模型，</p>
              <Button
                variant="outline"
                onClick={() => {
                  router.push('/config/model/llm');
                }}
              >
                前往添加
              </Button>
            </div>
          )}
        </div>
        <div className="px-2">
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="outline" className="text-xs">编辑提示词</Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-2xl lg:max-w-4xl max-h-[90vh] flex flex-col">
              <DialogHeader>
                <DialogTitle>编辑提示词</DialogTitle>
                <DialogDescription>
                  自定义 AI Agent 在不同阶段的行为提示词
                </DialogDescription>
              </DialogHeader>

              <div className="flex-1 overflow-hidden">
                <Tabs defaultValue="plan" className="h-full flex flex-col">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="plan">规划提示词</TabsTrigger>
                    <TabsTrigger value="act">行动提示词</TabsTrigger>
                    <TabsTrigger value="summarize">总结提示词</TabsTrigger>
                  </TabsList>

                  <div className="flex-1 overflow-hidden mt-4">
                    <TabsContent value="plan" className="h-full flex flex-col">
                      <ResettableTextarea
                        value={planPrompt}
                        onChange={(e) => setPlanPrompt(e.target.value)}
                        defaultValue={PLAN_PROMPT}
                        placeholder="输入规划阶段的提示词..."
                      />
                    </TabsContent>

                    <TabsContent value="act" className="h-full flex flex-col">
                      <ResettableTextarea
                        value={actPrompt}
                        onChange={(e) => setActPrompt(e.target.value)}
                        defaultValue={ACT_PROMPT}
                        placeholder="输入行动阶段的提示词..."
                      />
                    </TabsContent>

                    <TabsContent value="summarize" className="h-full flex flex-col">
                      <ResettableTextarea
                        value={summarizePrompt}
                        onChange={(e) => setSummarizePrompt(e.target.value)}
                        defaultValue={SUMMARY_PROMPT}
                        placeholder="输入总结阶段的提示词..."
                      />
                    </TabsContent>
                  </div>
                </Tabs>
              </div>

              <DialogFooter className="gap-2 sm:gap-0">
                <DialogClose asChild>
                  <Button variant="outline" onClick={
                    () => {
                      setActPrompt(botConfig.prompts.act);
                      setPlanPrompt(botConfig.prompts.plan);
                      setSummarizePrompt(botConfig.prompts.summary);
                    }
                  }>取消</Button>
                </DialogClose>
                <Button type="button" onClick={() => {
                  setBotConfig({
                    ...botConfig,
                    prompts: {
                      plan: planPrompt,
                      act: actPrompt,
                      summary: summarizePrompt,
                    }
                  });
                  // 关闭对话框
                  toast('success', { description: '提示词已保存' });
                }}>
                  保存更改
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>
      <div className="flex gap-6">
        <Label htmlFor="enable_search" className="w-[90px]">
          启用联网搜索
        </Label>
        <Switch
          id="enable_search"
          checked={botConfig.enable_search}
          onCheckedChange={(checked) => {
            setBotConfig({
              ...botConfig,
              enable_search: checked,
            });
          }}
        />
      </div>
      <div className="flex gap-6">
        <Label htmlFor="enable_agent" className="w-[90px]">
          Agentic模式
        </Label>
        <Switch
          id="enable_agent"
          checked={botConfig.enable_agent}
          onCheckedChange={(checked) => {
            setBotConfig({
              ...botConfig,
              enable_agent: checked,
            });
          }}
        />
      </div>
      <div className="flex">
        <Label htmlFor="kb_selection" className="w-[90px]">
          知识库选择
        </Label>
        <div className="pl-6 pr-6">
          {kbs.length > 0 ? (
            <DropdownMenu modal={true}>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  className="text-sm text-muted-foreground"
                >
                  已选{botConfig?.kb_ids.length || 0}个，可多选 <ChevronDownIcon />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>知识库</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {kbs.map((kb) => (
                  <DropdownMenuCheckboxItem
                    key={kb.id}
                    checked={botConfig.kb_ids.includes(kb.id)}
                    onCheckedChange={(checked) =>
                      handleKbSelect(kb.id, kb.name, checked)
                    }
                    onSelect={(e) => e.preventDefault()}
                  >
                    {kb.name}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <div>
              <p className="text-sm text-muted-foreground">尚未配置知识库</p>
            </div>
          )}
        </div>
        {selectedKbNames.length > 0 && (
          <div className="flex gap-1.5 items-center">
            {selectedKbNames.map((name) => (
              <Badge variant="secondary" className="h-6" key={name}>
                {name}
              </Badge>
            ))}
          </div>
        )}
      </div>
      <div className="flex">
        <Label htmlFor="mcp_selection" className="w-[90px]">
          MCP选择
        </Label>
        <div className="pl-6 pr-6">
          {mcps.length > 0 ? (
            <DropdownMenu modal={true}>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  className="text-sm text-muted-foreground"
                >
                  已选{botConfig.mcp_ids.length}个，可多选 <ChevronDownIcon />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>MCP</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {mcps.map((mcp) => (
                  <DropdownMenuCheckboxItem
                    key={mcp.id}
                    checked={botConfig.mcp_ids.includes(mcp.id)}
                    onCheckedChange={(checked) =>
                      handleMcpSelect(mcp.id, mcp.name, checked)
                    }
                    onSelect={(e) => e.preventDefault()}
                  >
                    {mcp.name}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <div>
              <p className="text-sm text-muted-foreground">尚未配置MCP</p>
            </div>
          )}
        </div>
        {selectedMcpNames.length > 0 && (
          <div className="flex gap-1.5 items-center">
            {selectedMcpNames.map((name) => (
              <Badge variant="secondary" className="h-6" key={name}>
                {name}
              </Badge>
            ))}
          </div>
        )}
      </div>
      <div className="flex items-center">
        <Label htmlFor="ai_guardrail" className="w-[90px]">
          AI安全护栏
        </Label>

        <div className="flex gap-4 pl-6 text-sm items-center">
          <div className="space-y-2">
            <Switch
              id="enable_input_check"
              checked={botConfig.enable_input_guardrail || false}
              onCheckedChange={(checked) => {
                setBotConfig({
                  ...botConfig,
                  enable_input_guardrail: checked,
                });
              }}
            />
            <Label htmlFor="input_guardrail" className="w-[120px]">
              输入护栏
            </Label>
          </div>
          <div className="space-y-2">
            <Switch
              id="enable_output_check"
              checked={botConfig.enable_output_guardrail || false}
              onCheckedChange={(checked) => {
                setBotConfig({
                  ...botConfig,
                  enable_output_guardrail: checked,
                });
              }}
            />
            <Label htmlFor="output_guardrail" className="w-[120px]">
              输出护栏
            </Label>
          </div>

          <div className="space-y-1">
            <Input
              className="w-120"
              value={botConfig.guardrail_hint || "作为人工智能助手，我无法回应包含不当或敏感信息的内容。"}
              onChange={(e) => {
                setBotConfig({
                  ...botConfig,
                  guardrail_hint: e.target.value,
                });

              }}
            />
            <Label htmlFor="guardrail_hint" className="w-[120px]">
              默认护栏提示
            </Label>
          </div>
        </div>
      </div>
      {saveErrorMsg && (
        <Alert variant="destructive">
          <Terminal />
          <AlertTitle>{isCreate ? '创建应用失败' : '保存应用失败'}</AlertTitle>
          <AlertDescription>{saveErrorMsg}</AlertDescription>
        </Alert>
      )}
      <div className="pt-6 flex gap-6">
        <Button
          variant="secondary"
          className="w-20"
          onClick={() => {
            router.push('/apps');
          }}
        >
          取消
        </Button>

        <Button
          className="w-20"
          onClick={() => {
            handleSaveChatConfig();
          }}
        >
          {isCreate ? '创建应用' : '保存应用'}
        </Button>
      </div>
    </div>
  );
};
