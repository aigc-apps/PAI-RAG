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
import { ChevronDownIcon } from "lucide-react";
import { Spinner } from "@/components/ui/loading";
import { useState, useEffect } from 'react';
import { McpConfig } from '@/app/config/mcp/mcp';
import { LlmConfig } from '@/app/config/model/llm/page';
import { KbConfig } from '@/app/knowledgebases/kbconfig';
import { useRouter } from 'next/navigation';
import { RunConfig } from '@/app/evaluation/[datasetId]/types';
import { ResettableTextarea } from '@/app/apps/resetable_textarea';
import { getPrompts } from '@/app/common/prompts';
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
  const [defaultReactPrompt, setDefaultReactPrompt] = useState('');
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
          react: ''
        },
      }
  );

  const [selectedKbNames, setSelectedKbNames] = useState<string[]>(
    kbs.filter(kb => localConfig.kb_ids.includes(kb.id)).map(kb => kb.name)
  );
  const [selectedMcpNames, setSelectedMcpNames] = useState<string[]>(
    mcps.filter(mcp => localConfig.mcp_ids.includes(mcp.id)).map(mcp => mcp.name)
  );

  const [reactPrompt, setReactPrompt] = useState('');
  const [openPrompt, setOpenPrompt] = useState(false);

  // Load default prompt from API on mount
  useEffect(() => {
    const loadDefaultPrompt = async () => {
      try {
        const prompts = await getPrompts();
        const defaultPrompt = prompts.react_prompt || '';
        console.log('Loaded default react prompt:', defaultPrompt);
        setDefaultReactPrompt(defaultPrompt);
        
        // Initialize localConfig with default prompt if in 'new' mode
        if (mode === 'new') {
          setLocalConfig(prev => ({
            ...prev,
            prompts: {
              react: defaultPrompt
            }
          }));
          setReactPrompt(defaultPrompt);
        }
      } catch (error) {
        console.error('Failed to load default prompt:', error);
      }
    };
    loadDefaultPrompt();
  }, [mode]);

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
      const prompt = config.prompts?.react || defaultReactPrompt;
      setReactPrompt(prompt);
    } else if (defaultReactPrompt) {
      const newConfig = {
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
          react: defaultReactPrompt,
        },
      };
      setLocalConfig(newConfig);
      setSelectedKbNames([]);
      setSelectedMcpNames([]);
      console.log("Setting react prompt (new mode):", defaultReactPrompt);
      setReactPrompt(defaultReactPrompt);
    }
  }, [mode, config, kbs, mcps, defaultReactPrompt]);

  // Sync reactPrompt when the prompt dialog opens
  useEffect(() => {
    if (openPrompt) {
      const currentPrompt = localConfig.prompts?.react || defaultReactPrompt;
      console.log("Syncing react prompt on dialog open:", currentPrompt);
      setReactPrompt(currentPrompt);
    }
  }, [openPrompt, localConfig.prompts, defaultReactPrompt]);

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

                    {/* Outer Tabs: Plan and Act blocks */}
                  <div className="flex-1 overflow-hidden">
                      <ResettableTextarea
                        value={reactPrompt}
                        onReset={() => setReactPrompt(defaultReactPrompt)}
                        onChange={(e) => setReactPrompt(e.target.value)}
                        defaultValue={defaultReactPrompt}
                      />
                  </div>

                  <DialogFooter className="gap-2 sm:gap-0">
                    <DialogClose asChild>
                      <Button variant="outline" onClick={() => {
                        setReactPrompt(localConfig.prompts.react);
                      }}>{t('common.cancel')}</Button>
                    </DialogClose>
                    <Button type="button" onClick={() => {
                      setLocalConfig((prev) => ({
                        ...prev,
                        prompts: {
                          react: reactPrompt,
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
                <Spinner size="sm" className="mr-2" />
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