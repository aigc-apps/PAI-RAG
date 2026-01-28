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
import { PLAN_PROMPT, ACT_PROMPT, ACT_WITH_PLAN_PROMPT, SUMMARY_PROMPT, getPrompts } from '../common/prompts';

// Add import for ResettableTextarea
import { ResettableTextarea } from '@/app/apps/resetable_textarea';


interface PromptConfig {
  plan: string;
  act: string;
  act_with_plan: string;
  summary: string;
}

interface FAQConfig {
  similarity_threshold?: number;
  embedding_model?: string;
  enable_question_in_retrieval?: boolean;
  enable_question_in_response?: boolean;
  enable_answer_in_retrieval?: boolean;
  enable_answer_in_response?: boolean;
  return_direct?: boolean;
}

export interface Chatbot {
  id: string;
  app_id: string;
  description: string;
  enable_search: boolean;
  enable_agent: boolean;
  enable_chatdb: boolean;
  enable_faq?: boolean;
  faq_config?: FAQConfig | null;
  mcp_ids: string[];
  kb_ids: string[];
  model_id: string;
  updated_at: string;
  enable_input_guardrail: boolean;
  enable_output_guardrail: boolean;
  guardrail_hint: string;
  prompts: PromptConfig;
}

// Props interface for controlled component
interface ChatbotConfigProps {
  botConfig: Chatbot;
  onConfigChange: (updates: Partial<Chatbot>) => void;
  onSave: () => Promise<boolean | void> | void;
  saving?: boolean;
  llms: LlmConfig[];
  mcps: McpConfig[];
  kbs: KbConfig[];
  isCreate?: boolean;
  saveErrorMsg?: string;
}

// 知识库配置卡片 - 受控组件
export const ChatbotConfigCard: FC<ChatbotConfigProps> = ({
  botConfig,
  onConfigChange,
  onSave,
  saving = false,
  llms,
  mcps,
  kbs,
  isCreate = false,
  saveErrorMsg: externalErrorMsg,
}) => {
  const [openPrompt, setOpenPrompt] = useState(false);
  const [selectedKbNames, setSelectedKbNames] = useState<string[]>([]);
  const [selectedMcpNames, setSelectedMcpNames] = useState<string[]>([]);
  const [saveErrorMsg, setSaveErrorMsg] = useState('');
  const [planPrompt, setPlanPrompt] = useState('');
  const [actPrompt, setActPrompt] = useState('');
  const [actWithPlanPrompt, setActWithPlanPrompt] = useState('');
  const [summarizePrompt, setSummarizePrompt] = useState('');
  const [defaultPrompts, setDefaultPrompts] = useState({
    plan: PLAN_PROMPT,
    act: ACT_PROMPT,
    act_with_plan: ACT_WITH_PLAN_PROMPT,
    summary: SUMMARY_PROMPT,
  });

  const router = useRouter();

  // Load default prompts from API (client-side)
  useEffect(() => {
    const loadDefaultPrompts = async () => {
      try {
        const prompts = await getPrompts();
        if (prompts) {
          const newDefaults = {
            plan: prompts.plan_prompt || PLAN_PROMPT,
            act: prompts.act_prompt || ACT_PROMPT,
            act_with_plan: prompts.act_with_plan_prompt || ACT_WITH_PLAN_PROMPT,
            summary: prompts.summary_prompt || SUMMARY_PROMPT,
          };
          setDefaultPrompts(newDefaults);

          if (isCreate) {
            const current = botConfig.prompts || {};
            const hasAnyPrompt = Boolean(
              (current.plan && current.plan.trim()) ||
              (current.act && current.act.trim()) ||
              (current.act_with_plan && current.act_with_plan.trim()) ||
              (current.summary && current.summary.trim())
            );
            if (!hasAnyPrompt) {
              onConfigChange({
                prompts: {
                  plan: newDefaults.plan,
                  act: newDefaults.act,
                  act_with_plan: newDefaults.act_with_plan,
                  summary: newDefaults.summary,
                },
              });
            }
          }
        }
      } catch (error) {
        console.error('Failed to load default prompts:', error);
      }
    };

    loadDefaultPrompts();
  }, [isCreate, botConfig.prompts, onConfigChange]);

  // Sync selected names when botConfig changes
  useEffect(() => {
    const kbnames = kbs
      .filter((item) => botConfig.kb_ids?.includes(item.id))
      .map((item) => item.name);
    setSelectedKbNames(kbnames);

    const mcpnames = mcps
      .filter((item) => botConfig.mcp_ids?.includes(item.id))
      .map((item) => item.name);
    setSelectedMcpNames(mcpnames);

    // Initialize prompts from botConfig
    setPlanPrompt(botConfig.prompts?.plan || defaultPrompts.plan);
    setActPrompt(botConfig.prompts?.act || defaultPrompts.act);
    setActWithPlanPrompt(botConfig.prompts?.act_with_plan || defaultPrompts.act_with_plan);
    setSummarizePrompt(botConfig.prompts?.summary || defaultPrompts.summary);
  }, [botConfig, kbs, mcps, defaultPrompts]);

  const handleKbSelect = (kb_id: string, kb_name: string, checked: boolean) => {
    if (checked) {
      const kb_ids = botConfig.kb_ids?.includes(kb_id)
        ? botConfig.kb_ids
        : [...(botConfig.kb_ids || []), kb_id];
      onConfigChange({ kb_ids });
      if (!selectedKbNames.includes(kb_name)) {
        setSelectedKbNames((prev) => [...prev, kb_name]);
      }
    } else {
      const kb_ids = (botConfig.kb_ids || []).filter((id) => id !== kb_id);
      onConfigChange({ kb_ids });
      if (selectedKbNames.includes(kb_name)) {
        setSelectedKbNames((prev) => prev.filter((name) => name !== kb_name));
      }
    }
  };

  const handleMcpSelect = (mcp_id: string, mcp_name: string, checked: boolean) => {
    if (checked) {
      const mcp_ids = botConfig.mcp_ids?.includes(mcp_id)
        ? botConfig.mcp_ids
        : [...(botConfig.mcp_ids || []), mcp_id];
      onConfigChange({ mcp_ids });
      if (!selectedMcpNames.includes(mcp_name)) {
        setSelectedMcpNames((prev) => [...prev, mcp_name]);
      }
    } else {
      const mcp_ids = (botConfig.mcp_ids || []).filter((id) => id !== mcp_id);
      onConfigChange({ mcp_ids });
      if (selectedMcpNames.includes(mcp_name)) {
        setSelectedMcpNames((prev) => prev.filter((name) => name !== mcp_name));
      }
    }
  };

  const handleSave = async () => {
    setSaveErrorMsg('');
    try {
      await onSave();
    } catch (err: any) {
      setSaveErrorMsg(err.message || '保存失败');
    }
  };

  const displayErrorMsg = externalErrorMsg || saveErrorMsg;

  return (
    <div className="grid gap-4 py-6 px-6">
      <div className="text-xl font-medium">
        {isCreate ? '新建应用' : '编辑应用'}
      </div>
      <div className="space-y-2">
        <Label htmlFor="app-id">
          App ID <span className="text-destructive">*</span>
        </Label>
        <Input
          id="appid"
          value={botConfig.app_id || ''}
          onChange={(e) => onConfigChange({ app_id: e.target.value })}
          placeholder="请输入应用ID, 如chatbot"
          required
          disabled={!isCreate}
        />
        <p className="text-sm text-muted-foreground">
          可输入大小写字母和数字,必须字母开头,3-64个字符。
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="description">描述</Label>
        <Textarea
          id="description"
          value={botConfig.description || ''}
          onChange={(e) => onConfigChange({ description: e.target.value })}
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
              value={botConfig.model_id || ''}
              onValueChange={(value) => onConfigChange({ model_id: value })}
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
              <p className="text-sm text-muted-foreground">尚未配置大模型</p>
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
          <Dialog open={openPrompt} onOpenChange={setOpenPrompt}>
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
                <Tabs defaultValue="plan_group" className="h-full flex flex-col">
                  <TabsList className="flex space-x-2">
                    <TabsTrigger value="plan_group">规划提示词</TabsTrigger>
                    <TabsTrigger value="act_group">行动提示词</TabsTrigger>
                  </TabsList>

                  <div className="flex-1 overflow-hidden mt-4">
                    <TabsContent value="plan_group" className="h-full flex flex-col">
                      <Tabs defaultValue="plan" className="h-full flex flex-col">
                        <TabsList className="grid grid-cols-3">
                          <TabsTrigger value="plan">规划</TabsTrigger>
                          <TabsTrigger value="act_with_plan">规划行动</TabsTrigger>
                          <TabsTrigger value="summary">规划总结</TabsTrigger>
                        </TabsList>
                        <div className="flex-1 overflow-hidden mt-2">
                          <TabsContent value="plan" className="h-full flex flex-col">
                            <ResettableTextarea
                              value={planPrompt}
                              onReset={() => setPlanPrompt(defaultPrompts.plan)}
                              onChange={(e) => setPlanPrompt(e.target.value)}
                              defaultValue={defaultPrompts.plan}
                              placeholder="输入规划阶段的提示词..."
                            />
                          </TabsContent>
                          <TabsContent value="act_with_plan" className="h-full flex flex-col">
                            <ResettableTextarea
                              value={actWithPlanPrompt}
                              onReset={() => setActWithPlanPrompt(defaultPrompts.act_with_plan)}
                              onChange={(e) => setActWithPlanPrompt(e.target.value)}
                              defaultValue={defaultPrompts.act_with_plan}
                              placeholder="输入规划驱动行动阶段的提示词..."
                            />
                          </TabsContent>
                          <TabsContent value="summary" className="h-full flex flex-col">
                            <ResettableTextarea
                              value={summarizePrompt}
                              onReset={() => setSummarizePrompt(defaultPrompts.summary)}
                              onChange={(e) => setSummarizePrompt(e.target.value)}
                              defaultValue={defaultPrompts.summary}
                              placeholder="输入总结阶段的提示词..."
                            />
                          </TabsContent>
                        </div>
                      </Tabs>
                    </TabsContent>

                    <TabsContent value="act_group" className="h-full flex flex-col">
                      <ResettableTextarea
                        value={actPrompt}
                        onReset={() => setActPrompt(defaultPrompts.act)}
                        onChange={(e) => setActPrompt(e.target.value)}
                        defaultValue={defaultPrompts.act}
                        placeholder="输入行动阶段的提示词..."
                      />
                    </TabsContent>
                  </div>
                </Tabs>
              </div>

              <DialogFooter className="gap-2 sm:gap-0">
                <DialogClose asChild>
                  <Button variant="outline" onClick={() => {
                    setActPrompt(botConfig.prompts?.act || defaultPrompts.act);
                    setPlanPrompt(botConfig.prompts?.plan || defaultPrompts.plan);
                    setActWithPlanPrompt(botConfig.prompts?.act_with_plan || defaultPrompts.act_with_plan);
                    setSummarizePrompt(botConfig.prompts?.summary || defaultPrompts.summary);
                  }}>取消</Button>
                </DialogClose>
                <Button type="button" onClick={() => {
                  onConfigChange({
                    prompts: {
                      plan: planPrompt,
                      act: actPrompt,
                      act_with_plan: actWithPlanPrompt,
                      summary: summarizePrompt,
                    }
                  });
                  setOpenPrompt(false);
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
          checked={botConfig.enable_search || false}
          onCheckedChange={(checked) => onConfigChange({ enable_search: checked })}
        />
      </div>
      <div className="flex gap-6">
        <Label htmlFor="enable_chatdb" className="w-[90px]">
          启用ChatDB
        </Label>
        <Switch
          id="enable_chatdb"
          checked={botConfig.enable_chatdb || false}
          onCheckedChange={(checked) => onConfigChange({ enable_chatdb: checked })}
        />
      </div>
      <div className="flex gap-6">
        <Label htmlFor="enable_agent" className="w-[90px]">
          Agentic模式
        </Label>
        <Switch
          id="enable_agent"
          checked={botConfig.enable_agent || false}
          onCheckedChange={(checked) => onConfigChange({ enable_agent: checked })}
        />
      </div>
      <div className="flex gap-6">
        <Label htmlFor="enable_faq" className="w-[90px]">
          启用FAQ
        </Label>
        <Switch
          id="enable_faq"
          checked={botConfig.enable_faq || false}
          onCheckedChange={(checked) => onConfigChange({ enable_faq: checked })}
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
                  已选{botConfig?.kb_ids?.length || 0}个，可多选 <ChevronDownIcon />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>知识库</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {kbs.map((kb) => (
                  <DropdownMenuCheckboxItem
                    key={kb.id}
                    checked={botConfig.kb_ids?.includes(kb.id) || false}
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
                  已选{botConfig.mcp_ids?.length || 0}个，可多选 <ChevronDownIcon />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>MCP</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {mcps.map((mcp) => (
                  <DropdownMenuCheckboxItem
                    key={mcp.id}
                    checked={botConfig.mcp_ids?.includes(mcp.id) || false}
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
              onCheckedChange={(checked) => onConfigChange({ enable_input_guardrail: checked })}
            />
            <Label htmlFor="input_guardrail" className="w-[120px]">
              输入护栏
            </Label>
          </div>
          <div className="space-y-2">
            <Switch
              id="enable_output_check"
              checked={botConfig.enable_output_guardrail || false}
              onCheckedChange={(checked) => onConfigChange({ enable_output_guardrail: checked })}
            />
            <Label htmlFor="output_guardrail" className="w-[120px]">
              输出护栏
            </Label>
          </div>

          <div className="space-y-1">
            <Input
              className="w-120"
              value={botConfig.guardrail_hint || "作为人工智能助手，我无法回应包含不当或敏感信息的内容。"}
              onChange={(e) => onConfigChange({ guardrail_hint: e.target.value })}
            />
            <Label htmlFor="guardrail_hint" className="w-[120px]">
              默认护栏提示
            </Label>
          </div>
        </div>
      </div>
      {displayErrorMsg && (
        <Alert variant="destructive">
          <Terminal />
          <AlertTitle>{isCreate ? '创建应用失败' : '保存应用失败'}</AlertTitle>
          <AlertDescription>{displayErrorMsg}</AlertDescription>
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
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? '保存中...' : (isCreate ? '创建应用' : '保存应用')}
        </Button>
      </div>
    </div>
  );
};
