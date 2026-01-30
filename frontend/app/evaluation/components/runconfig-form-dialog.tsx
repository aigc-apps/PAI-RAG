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
import { PLAN_PROMPT, ACT_PROMPT, ACT_WITH_PLAN_PROMPT, SUMMARY_PROMPT } from '@/app/common/prompts';
import { set } from "date-fns";
import { useI18n } from '@/app/providers/i18n';


interface RunConfigFormDialogProps {
  mode: 'new' | 'edit';
  config?: RunConfig; // When editing, pass in config
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
  const { t } = useI18n();
  const [localConfig, setLocalConfig] = useState<RunConfig>(
    mode === 'edit' && config
      ? { ...config }
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
        guardrail_hint: t('apps.guardrailHint'),
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

  const [planPrompt, setPlanPrompt] = useState('');
  const [actPrompt, setActPrompt] = useState('');
  const [actWithPlanPrompt, setActWithPlanPrompt] = useState('');
  const [summarizePrompt, setSummarizePrompt] = useState('');
  const [openPrompt, setOpenPrompt] = useState(false);

  // Reset form when config or mode changes
  useEffect(() => {
    if (mode === 'edit' && config) {
      setLocalConfig({ ...config });
      setSelectedKbNames(
        kbs.filter(kb => config.kb_ids.includes(kb.id)).map(kb => kb.name)
      );
      setSelectedMcpNames(
        mcps.filter(mcp => config.mcp_ids.includes(mcp.id)).map(mcp => mcp.name)
      );
      setPlanPrompt(config.prompts.plan);
      setActPrompt(config.prompts.act);
      setActWithPlanPrompt(config.prompts.act_with_plan);
      setSummarizePrompt(config.prompts.summary);
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
        guardrail_hint: t('apps.guardrailHint'),
        prompts: {
          plan: PLAN_PROMPT,
          act: ACT_PROMPT,
          act_with_plan: ACT_WITH_PLAN_PROMPT,
          summary: SUMMARY_PROMPT,
        },
      });
      setSelectedKbNames([]);
      setSelectedMcpNames([]);
    }
  }, [mode, config, kbs, mcps]);

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

  const mode_str = mode === "new" ? t('common.create') : t('common.edit');
  const titleKey = mode === "new" ? 'evaluation.newRunConfig' : 'evaluation.editRunConfig';
  const descKey = mode === "new" ? 'evaluation.newRunConfigDesc' : 'evaluation.editRunConfigDesc';

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t(titleKey)}</DialogTitle>
          <DialogDescription>{t(descKey)}</DialogDescription>
        </DialogHeader>

        <div className="grid gap-6 py-4">
          {/* Name */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="setting_name">{t('evaluation.configName')}</Label>
            <Input
              id="setting_name"
              value={localConfig.name}
              onChange={(e) =>
                setLocalConfig((prev) => ({ ...prev, name: e.target.value }))
              }
              placeholder={t('evaluation.configNamePlaceholder')}
              required
            />
          </div>

          {/* Base model selection */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="basemodel" className="flex items-center">
              {t('evaluation.baseModelSelection')} <span className="text-destructive ml-1">*</span>
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
                    <SelectValue placeholder={t('evaluation.selectBaseModel')} />
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
                  <p className="text-sm text-muted-foreground">{t('evaluation.noLlmConfigured')}</p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => router.push('/config/model/llm')}
                  >
                    {t('evaluation.goToAdd')}
                  </Button>
                </div>
              )}
            </div>
          </div>
          {/* Prompt settings */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label className="flex items-center">
              {t('evaluation.promptSettings')}
            </Label>
            <div className="px-2">
              <Dialog open={openPrompt} onOpenChange={setOpenPrompt}>
                <DialogTrigger asChild>
                  <Button variant="outline" className="text-xs">{t('evaluation.editPrompts')}</Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-2xl lg:max-w-4xl max-h-[90vh] flex flex-col">
                  <DialogHeader>
                    <DialogTitle>{t('evaluation.editPrompts')}</DialogTitle>
                    <DialogDescription>
                      {t('evaluation.customizeAgentPrompts')}
                    </DialogDescription>
                  </DialogHeader>

                  <div className="flex-1 overflow-hidden">
                    {/* Outer Tabs: Plan and Act blocks */}
                    <Tabs defaultValue="plan_group" className="h-full flex flex-col">
                      <TabsList className="flex space-x-2">
                        <TabsTrigger value="plan_group">{t('evaluation.planningPrompts')}</TabsTrigger>
                        <TabsTrigger value="act_group">{t('evaluation.actionPrompts')}</TabsTrigger>
                      </TabsList>

                      <div className="flex-1 overflow-hidden mt-4">
                        {/* Plan block content: 3 sub-tabs inside */}
                        <TabsContent value="plan_group" className="h-full flex flex-col">
                          <Tabs defaultValue="plan" className="h-full flex flex-col">
                            <TabsList className="grid grid-cols-3">
                              <TabsTrigger value="plan">{t('evaluation.planning')}</TabsTrigger>
                              <TabsTrigger value="act_with_plan">{t('evaluation.planningAction')}</TabsTrigger>
                              <TabsTrigger value="summary">{t('evaluation.planningSummary')}</TabsTrigger>
                            </TabsList>
                            <div className="flex-1 overflow-hidden mt-2">
                              <TabsContent value="plan" className="h-full flex flex-col">
                                <ResettableTextarea
                                  value={planPrompt}
                                  onReset={() => setPlanPrompt(PLAN_PROMPT)}
                                  onChange={(e) => setPlanPrompt(e.target.value)}
                                  defaultValue={PLAN_PROMPT}
                                  placeholder={t('evaluation.planningPromptPlaceholder')}
                                />
                              </TabsContent>
                              <TabsContent value="act_with_plan" className="h-full flex flex-col">
                                <ResettableTextarea
                                  value={actWithPlanPrompt}
                                  onReset={() => setActWithPlanPrompt(ACT_WITH_PLAN_PROMPT)}
                                  onChange={(e) => setActWithPlanPrompt(e.target.value)}
                                  defaultValue={ACT_WITH_PLAN_PROMPT}
                                  placeholder={t('evaluation.planActionPromptPlaceholder')}
                                />
                              </TabsContent>
                              <TabsContent value="summary" className="h-full flex flex-col">
                                <ResettableTextarea
                                  value={summarizePrompt}
                                  onReset={() => setSummarizePrompt(SUMMARY_PROMPT)}
                                  onChange={(e) => setSummarizePrompt(e.target.value)}
                                  defaultValue={SUMMARY_PROMPT}
                                  placeholder={t('evaluation.summaryPromptPlaceholder')}
                                />
                              </TabsContent>
                            </div>
                          </Tabs>
                        </TabsContent>

                        {/* Act block content: single Textarea */}
                        <TabsContent value="act_group" className="h-full flex flex-col">
                          <ResettableTextarea
                            value={actPrompt}
                            onReset={() => setActPrompt(ACT_PROMPT)}
                            onChange={(e) => setActPrompt(e.target.value)}
                            defaultValue={ACT_PROMPT}
                            placeholder={t('evaluation.actionPromptPlaceholder')}
                          />
                        </TabsContent>
                      </div>
                    </Tabs>
                  </div>

                  <DialogFooter className="gap-2 sm:gap-0">
                    <DialogClose asChild>
                      <Button variant="outline" onClick={() => {
                        setActPrompt(localConfig.prompts.act);
                        setPlanPrompt(localConfig.prompts.plan);
                        setActWithPlanPrompt(localConfig.prompts.act_with_plan);
                        setSummarizePrompt(localConfig.prompts.summary);
                      }}>{t('common.cancel')}</Button>
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
                      {t('evaluation.saveChanges')}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </div>
          {/* Enable web search */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="enable_search">{t('evaluation.enableWebSearch')}</Label>
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

          {/* Agentic mode */}
          <div className="grid grid-cols-[120px_1fr] items-center gap-4">
            <Label htmlFor="enable_agent">{t('evaluation.agenticMode')}</Label>
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

          {/* Knowledge base selection */}
          <div className="grid grid-cols-[120px_1fr] items-start gap-4">
            <Label htmlFor="kb_selection">{t('evaluation.kbSelection')}</Label>
            <div className="space-y-2">
              {kbs.length > 0 ? (
                <DropdownMenu modal={true}>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      className="text-sm text-muted-foreground"
                    >
                      {t('evaluation.selectedCount', { count: localConfig?.kb_ids.length || 0 })} <ChevronDownIcon />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-56">
                    <DropdownMenuLabel>{t('evaluation.knowledgebase')}</DropdownMenuLabel>
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
                <p className="text-sm text-muted-foreground">{t('evaluation.noKbConfigured')}</p>
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

          {/* MCP selection */}
          <div className="grid grid-cols-[120px_1fr] items-start gap-4">
            <Label htmlFor="mcp_selection">{t('evaluation.mcpSelection')}</Label>
            <div className="space-y-2">
              {mcps.length > 0 ? (
                <DropdownMenu modal={true}>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      className="text-sm text-muted-foreground"
                    >
                      {t('evaluation.selectedCount', { count: localConfig.mcp_ids.length })} <ChevronDownIcon />
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
                <p className="text-sm text-muted-foreground">{t('evaluation.noMcpConfigured')}</p>
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

          {/* AI Guardrail */}
          <div className="grid grid-cols-[120px_1fr] items-start gap-4">
            <Label>{t('evaluation.aiGuardrail')}</Label>
            <div className="space-y-4">
              {/* Input/output guardrail */}
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
                    {t('evaluation.inputGuardrailLabel')}
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
                    {t('evaluation.outputGuardrailLabel')}
                  </Label>
                </div>
              </div>

              {/* Default guardrail hint */}
              <div className="space-y-1">
                <Input
                  id="guardrail_hint"
                  placeholder={t('apps.guardrailHint')}
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
                  {t('apps.guardrailHintTip')}
                </Label>
              </div>
            </div>
          </div>
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <DialogClose asChild>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              {t('common.cancel')}
            </Button>
          </DialogClose>


          <Button onClick={handleSubmit} disabled={isSaving}>
            {isSaving ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t('common.saving')}
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