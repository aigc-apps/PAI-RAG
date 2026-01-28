// components/EvalConfigFormDialog.tsx
'use client';

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogTrigger,
  DialogClose
} from "@/components/ui/dialog";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Switch } from '@/components/ui/switch';
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Loader2, ChevronDownIcon } from "lucide-react";
import { useState, useEffect } from 'react';
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';
import { useRouter } from 'next/navigation';
import { RunConfig } from '@/app/evaluation/[datasetId]/types';
import { ResettableTextarea } from '@/app/apps/resetable_textarea';
import { PLAN_PROMPT, ACT_PROMPT, ACT_WITH_PLAN_PROMPT, SUMMARY_PROMPT, getPrompts } from '@/app/common/prompts';
import { set } from "date-fns";


interface RunConfigFormDialogProps {
  mode: 'new' | 'edit';
  config?: RunConfig; // edit 时传入
  llms: LlmConfig[];
  mcps: McpConfig[];
  kbs: KbConfig[];
  datasetId: string;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (config: RunConfig) => void;
  isSaving: boolean;
}

export function RunConfigFormDialog({
  mode,
  config,
  llms,
  mcps,
  kbs,
  datasetId,
  isOpen,
  onOpenChange,
  onSave,
  isSaving,
}: RunConfigFormDialogProps) {
  const router = useRouter();
  const [localConfig, setLocalConfig] = useState<RunConfig>(
    mode === 'edit' && config
      ? {
          ...config,
          // 确保 prompts 字段存在且有默认值
          prompts: {
            plan: config.prompts?.plan || PLAN_PROMPT,
            act: config.prompts?.act || ACT_PROMPT,
            act_with_plan: config.prompts?.act_with_plan || ACT_WITH_PLAN_PROMPT,
            summary: config.prompts?.summary || SUMMARY_PROMPT,
          },
          // 确保 parallel_count 有默认值
          parallel_count: config.parallel_count || 1,
        }
      : {
        id: "",
        name: "",
        model_id: "",
        mcp_ids: [],
        kb_ids: [],
        enable_search: false,
        enable_vision: false,
        enable_agent: false,
        enable_input_guardrail: false,
        enable_output_guardrail: false,
        guardrail_hint: "作为人工智能助手，我无法回应包含不当或敏感信息的内容。",
        parallel_count: 1,
        prompts: {
          plan: PLAN_PROMPT,
          act: ACT_PROMPT,
          act_with_plan: ACT_WITH_PLAN_PROMPT,
          summary: SUMMARY_PROMPT,
        },
      }
  );

  const [selectedKbNames, setSelectedKbNames] = useState<string[]>(
    kbs.filter(kb => localConfig.kb_ids.includes(kb.id)).map(kb => kb.name)
  );
  const [selectedMcpNames, setSelectedMcpNames] = useState<string[]>(
    mcps.filter(mcp => localConfig.mcp_ids.includes(mcp.id)).map(mcp => mcp.name)
  );

  const [planPrompt, setPlanPrompt] = useState(PLAN_PROMPT);
  const [actPrompt, setActPrompt] = useState(ACT_PROMPT);
  const [actWithPlanPrompt, setActWithPlanPrompt] = useState(ACT_WITH_PLAN_PROMPT);
  const [summarizePrompt, setSummarizePrompt] = useState(SUMMARY_PROMPT);
  const [openPrompt, setOpenPrompt] = useState(false);
  const [defaultPrompts, setDefaultPrompts] = useState({
    plan: PLAN_PROMPT,
    act: ACT_PROMPT,
    act_with_plan: ACT_WITH_PLAN_PROMPT,
    summary: SUMMARY_PROMPT,
  });

  // 从 API 加载默认提示词
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
          
          // 如果当前是创建模式且提示词为空，则使用从 API 获取的默认值
          if (mode === 'new' && !config) {
            setPlanPrompt(newDefaults.plan);
            setActPrompt(newDefaults.act);
            setActWithPlanPrompt(newDefaults.act_with_plan);
            setSummarizePrompt(newDefaults.summary);
            // 同时更新 localConfig 中的 prompts
            setLocalConfig(prev => ({
              ...prev,
              prompts: {
                plan: newDefaults.plan,
                act: newDefaults.act,
                act_with_plan: newDefaults.act_with_plan,
                summary: newDefaults.summary,
              },
            }));
          }
        }
      } catch (error) {
        console.error('Failed to load default prompts:', error);
        // 如果加载失败，使用导入的常量（可能是空字符串，但至少不会报错）
      }
    };
    
    loadDefaultPrompts();
  }, []); // 只在组件挂载时执行一次

  // 当 config 或 mode 变化时重置表单
  useEffect(() => {
    if (mode === 'edit' && config) {
      setLocalConfig({ 
        ...config,
        parallel_count: config.parallel_count || 1,
      });
      setSelectedKbNames(
        kbs.filter(kb => config.kb_ids.includes(kb.id)).map(kb => kb.name)
      );
      setSelectedMcpNames(
        mcps.filter(mcp => config.mcp_ids.includes(mcp.id)).map(mcp => mcp.name)
      );
          // 安全地读取提示词，如果不存在则使用默认值
          const prompts = config.prompts || {};
          setPlanPrompt(prompts.plan || defaultPrompts.plan);
          setActPrompt(prompts.act || defaultPrompts.act);
          setActWithPlanPrompt(prompts.act_with_plan || defaultPrompts.act_with_plan);
          setSummarizePrompt(prompts.summary || defaultPrompts.summary);
    } else {
      setLocalConfig({
        id: "",
        name: "",
        model_id: "",
        mcp_ids: [],
        kb_ids: [],
        enable_search: false,
        enable_vision: false,
        enable_agent: false,
        enable_input_guardrail: false,
        enable_output_guardrail: false,
        guardrail_hint: "作为人工智能助手，我无法回应包含不当或敏感信息的内容。",
        parallel_count: 1,
        prompts: {
          plan: PLAN_PROMPT,
          act: ACT_PROMPT,
          act_with_plan: ACT_WITH_PLAN_PROMPT,
          summary: SUMMARY_PROMPT,
        },
      });
      setSelectedKbNames([]);
      setSelectedMcpNames([]);
      // 初始化提示词为默认值（使用从 API 加载的默认值）
      setPlanPrompt(defaultPrompts.plan);
      setActPrompt(defaultPrompts.act);
      setActWithPlanPrompt(defaultPrompts.act_with_plan);
      setSummarizePrompt(defaultPrompts.summary);
    }
  }, [mode, config, kbs, mcps, defaultPrompts]);

  const handleKbSelect = (kb_id: string, kb_name: string, checked: boolean) => {
    setLocalConfig(prev => {
      const kb_ids = checked
        ? prev.kb_ids.includes(kb_id) ? prev.kb_ids : [...prev.kb_ids, kb_id]
        : prev.kb_ids.filter(id => id !== kb_id);
      return { ...prev, kb_ids };
    });

    setSelectedKbNames(prev => {
      if (checked) {
        return prev.includes(kb_name) ? prev : [...prev, kb_name];
      } else {
        return prev.filter(name => name !== kb_name);
      }
    });
  };

  const handleMcpSelect = (mcp_id: string, mcp_name: string, checked: boolean) => {
    setLocalConfig(prev => {
      const mcp_ids = checked
        ? prev.mcp_ids.includes(mcp_id) ? prev.mcp_ids : [...prev.mcp_ids, mcp_id]
        : prev.mcp_ids.filter(id => id !== mcp_id);
      return { ...prev, mcp_ids };
    });

    setSelectedMcpNames(prev => {
      if (checked) {
        return prev.includes(mcp_name) ? prev : [...prev, mcp_name];
      } else {
        return prev.filter(name => name !== mcp_name);
      }
    });
  };

  const handleSubmit = () => {
    onSave(localConfig);
  };

  const mode_str = mode === "new" ? "新建" : "修改";

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{mode_str}运行配置</DialogTitle>
          <DialogDescription>{mode_str}一个运行配置后，点击保存。</DialogDescription>
        </DialogHeader>

        <div className="grid gap-6 py-4">
          {/* 名称 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="setting_name">配置名称</Label>
            <Input
              id="setting_name"
              value={localConfig.name}
              onChange={(e) =>
                setLocalConfig((prev) => ({ ...prev, name: e.target.value }))
              }
              placeholder="请输入配置名称, 如config_v1"
              required
            />
          </div>

          {/* 基模型选择 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="basemodel" className="flex items-center">
              基模型选择 <span className="text-destructive ml-1">*</span>
            </Label>
            <div>
              {llms.length > 0 ? (
                <Select
                  value={localConfig.model_id}
                  onValueChange={(value) =>
                    setLocalConfig((prev) => ({
                      ...prev,
                      model_id: value,
                    }))
                  }
                >
                  <SelectTrigger id="basemodel">
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
                <div className="flex flex-col gap-2">
                  <p className="text-sm text-muted-foreground">尚未配置大模型</p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => router.push('/config/model/llm')}
                  >
                    前往添加
                  </Button>
                </div>
              )}
            </div>
          </div>
          {/* 提示词设置 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label className="flex items-center">
              提示词设置
            </Label>
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
                    {/* 外层 Tabs：分 Plan、Act 两大块 */}
                    <Tabs defaultValue="plan_group" className="h-full flex flex-col">
                      <TabsList className="flex space-x-2">
                        <TabsTrigger value="plan_group">规划提示词</TabsTrigger>
                        <TabsTrigger value="act_group">行动提示词</TabsTrigger>
                      </TabsList>

                      <div className="flex-1 overflow-hidden mt-4">
                        {/* Plan 块内容：内部再分 3 个子 Tab */}
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

                        {/* Act 块内容：单独一个 Textarea */}
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
                        // 安全地读取提示词，如果不存在则使用默认值
                        const prompts = localConfig.prompts || {};
                        setActPrompt(prompts.act || defaultPrompts.act);
                        setPlanPrompt(prompts.plan || defaultPrompts.plan);
                        setActWithPlanPrompt(prompts.act_with_plan || defaultPrompts.act_with_plan);
                        setSummarizePrompt(prompts.summary || defaultPrompts.summary);
                      }}>取消</Button>
                    </DialogClose>
                    <Button type="button" onClick={() => {
                      setLocalConfig((prev) => ({
                        ...prev,
                        prompts: {
                          plan: planPrompt,
                          act: actPrompt,
                          act_with_plan: actWithPlanPrompt,
                          summary: summarizePrompt,
                        }
                      }));
                      setOpenPrompt(false);
                    }}>
                      保存更改
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </div>
          {/* 启用联网搜索 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="enable_search">启用联网搜索</Label>
            <Switch
              id="enable_search"
              className="justify-self-start"
              checked={localConfig.enable_search}
              onCheckedChange={(checked) => {
                setLocalConfig((prev) => ({
                  ...prev,
                  enable_search: checked,
                }));
              }}
            />
          </div>

          {/* Agentic模式 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="enable_agent">Agentic模式</Label>
            <Switch
              id="enable_agent"
              className="justify-self-start"
              checked={localConfig.enable_agent}
              onCheckedChange={(checked) => {
                setLocalConfig((prev) => ({
                  ...prev,
                  enable_agent: checked,
                }));
              }}
            />
          </div>

          {/* 知识库选择 */}
          <div className="grid grid-cols-[120px_1fr] items-start gap-4">
            <Label htmlFor="kb_selection">知识库选择</Label>
            <div className="space-y-2">
              {kbs.length > 0 ? (
                <DropdownMenu modal={true}>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      className="text-sm text-muted-foreground"
                    >
                      已选{localConfig?.kb_ids.length || 0}个，可多选 <ChevronDownIcon />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-56">
                    <DropdownMenuLabel>知识库</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    {kbs.map((kb) => (
                      <DropdownMenuCheckboxItem
                        key={kb.id}
                        checked={localConfig.kb_ids.includes(kb.id)}
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
                <p className="text-sm text-muted-foreground">尚未配置知识库</p>
              )}

              {selectedKbNames.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {selectedKbNames.map((name) => (
                    <Badge variant="secondary" key={name}>
                      {name}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* MCP选择 */}
          <div className="grid grid-cols-[120px_1fr] items-start gap-4">
            <Label htmlFor="mcp_selection">MCP选择</Label>
            <div className="space-y-2">
              {mcps.length > 0 ? (
                <DropdownMenu modal={true}>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      className="text-sm text-muted-foreground"
                    >
                      已选{localConfig.mcp_ids.length}个，可多选 <ChevronDownIcon />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-56">
                    <DropdownMenuLabel>MCP</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    {mcps.map((mcp) => (
                      <DropdownMenuCheckboxItem
                        key={mcp.id}
                        checked={localConfig.mcp_ids.includes(mcp.id)}
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
                <p className="text-sm text-muted-foreground">尚未配置MCP</p>
              )}

              {selectedMcpNames.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {selectedMcpNames.map((name) => (
                    <Badge variant="secondary" key={name}>
                      {name}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* 任务并行数 */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="parallel_count">任务并行数</Label>
            <div className="space-y-1">
              <Input
                id="parallel_count"
                type="number"
                min="1"
                max="50"
                value={localConfig.parallel_count || 1}
                onChange={(e) => {
                  const value = parseInt(e.target.value, 10);
                  setLocalConfig((prev) => ({
                    ...prev,
                    parallel_count: isNaN(value) || value < 1 ? 1 : value,
                  }));
                }}
                placeholder="请输入并行任务数，默认为1"
              />
              <Label htmlFor="parallel_count" className="text-xs text-muted-foreground">
                设置评估实验的并行任务数，建议值：1-10
              </Label>
            </div>
          </div>

          {/* AI安全护栏 */}
          <div className="grid grid-cols-[120px_1fr] items-start gap-4">
            <Label>AI安全护栏</Label>
            <div className="space-y-4">
              {/* 输入/输出护栏 */}
              <div className="grid grid-cols-2 gap-6">
                <div className="flex items-center gap-2">
                  <Switch
                    id="enable_input_check"
                    checked={localConfig.enable_input_guardrail || false}
                    onCheckedChange={(checked) => {
                      setLocalConfig((prev) => ({
                        ...prev,
                        enable_input_guardrail: checked,
                      }));
                    }}
                  />
                  <Label htmlFor="enable_input_check" className="text-sm">
                    输入护栏
                  </Label>
                </div>
                <div className="flex items-center gap-2">
                  <Switch
                    id="enable_output_check"
                    checked={localConfig.enable_output_guardrail || false}
                    onCheckedChange={(checked) => {
                      setLocalConfig((prev) => ({
                        ...prev,
                        enable_output_guardrail: checked,
                      }));
                    }}
                  />
                  <Label htmlFor="enable_output_check" className="text-sm">
                    输出护栏
                  </Label>
                </div>
              </div>

              {/* 默认护栏提示 */}
              <div className="space-y-1">
                <Input
                  id="guardrail_hint"
                  placeholder="作为人工智能助手，我无法回应包含不当或敏感信息的内容。"
                  className="w-full"
                  value={localConfig.guardrail_hint || ""}
                  onChange={(e) => {
                    setLocalConfig((prev) => ({
                      ...prev,
                      guardrail_hint: e.target.value,
                    }));
                  }}
                />
                <Label htmlFor="guardrail_hint" className="text-xs text-muted-foreground">
                  默认护栏提示
                </Label>
              </div>
            </div>
          </div>
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <DialogClose asChild>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              取消
            </Button>
          </DialogClose>


          <Button onClick={handleSubmit} disabled={isSaving}>
            {isSaving ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                提交中...
              </>
            ) : (
              mode_str
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}