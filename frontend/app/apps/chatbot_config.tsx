'use client';
import React, { useState, useEffect, FC } from 'react';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { ChevronDownIcon } from 'lucide-react';
import { useI18n } from '@/app/providers/i18n';

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
import { REACT_PROMPT, getPrompts } from '../common/prompts';

// Add import for ResettableTextarea
import { ResettableTextarea } from '@/app/apps/resetable_textarea';
import { toast } from 'sonner';


interface PromptConfig {
  react: string;
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
  enable_auto_metadata_filter?: boolean;
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
  const { t } = useI18n();
  const [openPrompt, setOpenPrompt] = useState(false);
  const [selectedKbNames, setSelectedKbNames] = useState<string[]>([]);
  const [selectedMcpNames, setSelectedMcpNames] = useState<string[]>([]);
  const [saveErrorMsg, setSaveErrorMsg] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [defaultPrompts, setDefaultPrompts] = useState({
    react: REACT_PROMPT,
  });

  const router = useRouter();

  // Load default prompts from API (client-side)
  useEffect(() => {
    const loadDefaultPrompts = async () => {
      try {
        const prompts = await getPrompts();
        if (prompts) {
          const newDefaults = {
            react: prompts.react_prompt || REACT_PROMPT,
          };
          setDefaultPrompts(newDefaults);

          if (isCreate) {
            const current = botConfig.prompts || {};
            const hasAnyPrompt = Boolean(
              (current.react && current.react.trim())
            );
            if (!hasAnyPrompt) {
              onConfigChange({
                prompts: {
                  react: newDefaults.react,
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
    setSystemPrompt(botConfig.prompts?.react || defaultPrompts.react);
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
      toast.error(err.message || t('messages.saveError'));
    }
  };


  return (
    <div className="space-y-4 px-6 pt-3">
      <div className="space-y-2">
        <Label htmlFor="app-id">
          App ID <span className="text-destructive">*</span>
        </Label>
        <Input
          id="appid"
          value={botConfig.app_id || ''}
          onChange={(e) => onConfigChange({ app_id: e.target.value })}
          placeholder={t('apps.appIdPlaceholder')}
          required
          disabled={!isCreate}
        />
        <p className="text-sm text-muted-foreground">
          {t('apps.appIdTip')}
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="description">{t('apps.descriptionLabel')}</Label>
        <Textarea
          id="description"
          value={botConfig.description || ''}
          onChange={(e) => onConfigChange({ description: e.target.value })}
          placeholder={t('apps.descriptionPlaceholder')}
          rows={3}
        />
      </div>
      <div className="flex">
        <Label htmlFor="basemodel" className="w-[120px]">
          {t('apps.baseModel')} <span className="text-destructive">*</span>{' '}
        </Label>
        <div className="px-6">
          {llms.length > 0 ? (
            <Select
              value={botConfig.model_id || ''}
              onValueChange={(value) => onConfigChange({ model_id: value })}
            >
              <SelectTrigger>
                <SelectValue placeholder={t('apps.selectModel')} />
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
              <p className="text-sm text-muted-foreground">{t('apps.noModelConfigured')}</p>
              <Button
                variant="outline"
                onClick={() => {
                  router.push('/config/model/llm');
                }}
              >
                {t('apps.addModel')}
              </Button>
            </div>
          )}
        </div>
        <div className="px-2">
          <Dialog open={openPrompt} onOpenChange={setOpenPrompt}>
            <DialogTrigger asChild>
              <Button variant="outline" className="text-xs">{t('apps.editPrompts')}</Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-2xl lg:max-w-4xl max-h-[90vh] flex flex-col">
              <DialogHeader>
                <DialogTitle>{t('apps.editPrompts')}</DialogTitle>
                <DialogDescription>
                  {t('apps.editPromptDesc')}
                </DialogDescription>
              </DialogHeader>

              <div className="flex-1 overflow-hidden">
                  <div className="flex-1 overflow-hidden">
                      <ResettableTextarea
                        value={systemPrompt}
                        onReset={() => setSystemPrompt(defaultPrompts.react)}
                        onChange={(e) => setSystemPrompt(e.target.value)}
                        defaultValue={defaultPrompts.react}
                      />
                  </div>
              </div>

              <DialogFooter className="gap-2 sm:gap-0">
                <div className="flex items-center justify-between w-full">
                  <p className="text-sm text-muted-foreground" suppressHydrationWarning>
                    {t('apps.promptSaveReminder')}
                  </p>
                  <div className="flex gap-2">
                    <DialogClose asChild>
                      <Button variant="outline" onClick={() => {
                        setSystemPrompt(botConfig.prompts?.react || defaultPrompts.react);
                      }}>{t('common.cancel')}</Button>
                    </DialogClose>
                    <Button type="button" onClick={async () => {
                      onConfigChange({
                        prompts: {
                          react: systemPrompt,
                        }
                      });
                      setOpenPrompt(false);
                    }}>
                      {t('common.save')}
                    </Button>
                  </div>
                </div>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>
      <div className="flex gap-6">
        <Label htmlFor="enable_search" className="w-[120px]">
          {t('apps.enableSearch')}
        </Label>
        <Switch
          id="enable_search"
          checked={botConfig.enable_search || false}
          onCheckedChange={(checked) => onConfigChange({ enable_search: checked })}
        />
      </div>
      <div className="flex gap-6">
        <Label htmlFor="enable_chatdb" className="w-[120px]">
          {t('apps.enableChatDb')}
        </Label>
        <Switch
          id="enable_chatdb"
          checked={botConfig.enable_chatdb || false}
          onCheckedChange={(checked) => onConfigChange({ enable_chatdb: checked })}
        />
      </div>
      <div className="flex gap-6">
        <Label htmlFor="enable_faq" className="w-[120px]">
          {t('apps.enableFaq')}
        </Label>
        <Switch
          id="enable_faq"
          checked={botConfig.enable_faq || false}
          onCheckedChange={(checked) => onConfigChange({ enable_faq: checked })}
        />
      </div>
      <div className="flex">
        <Label htmlFor="kb_selection" className="w-[120px]">
          {t('apps.knowledgebaseSelection')}
        </Label>
        <div className="pl-6 pr-6">
          {kbs.length > 0 ? (
            <DropdownMenu modal={true}>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  className="text-sm text-muted-foreground"
                >
                  {t('apps.selectedKbNum', { num: botConfig?.kb_ids?.length || 0 })} <ChevronDownIcon />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56">
                <DropdownMenuLabel>{t('apps.knowledgebaseSelection')}</DropdownMenuLabel>
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
              <p className="text-sm text-muted-foreground">{t('apps.noKbConfigured')}</p>
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
      <div className="flex gap-6">
        <Label htmlFor="enable_auto_metadata_filter" className="w-[120px]">
          {t('apps.enableAutoMetadataFilter')}
        </Label>
        <Switch
          id="enable_auto_metadata_filter"
          checked={botConfig.enable_auto_metadata_filter || false}
          onCheckedChange={(checked) => onConfigChange({ enable_auto_metadata_filter: checked })}
        />
      </div>
      <div className="flex">
        <Label htmlFor="mcp_selection" className="w-[120px]">
          {t('apps.mcpSelection')}
        </Label>
        <div className="pl-6 pr-6">
          {mcps.length > 0 ? (
            <DropdownMenu modal={true}>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  className="text-sm text-muted-foreground"
                >
                  {t('apps.selectedMcpNum', { num: botConfig.mcp_ids?.length || 0 })}
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
              <p className="text-sm text-muted-foreground">{t('apps.noMcpConfigured')}</p>
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
        <Label htmlFor="ai_guardrail" className="w-[120px]">
          {t('apps.guardrail')}
        </Label>

        <div className="flex gap-4 pl-6 text-sm items-center">
          <div className="space-y-2">
            <Switch
              id="enable_input_check"
              checked={botConfig.enable_input_guardrail || false}
              onCheckedChange={(checked) => onConfigChange({ enable_input_guardrail: checked })}
            />
            <Label htmlFor="input_guardrail" className="w-[120px]">
              {t('apps.inputGuardrail')}
            </Label>
          </div>
          <div className="space-y-2">
            <Switch
              id="enable_output_check"
              checked={botConfig.enable_output_guardrail || false}
              onCheckedChange={(checked) => onConfigChange({ enable_output_guardrail: checked })}
            />
            <Label htmlFor="output_guardrail" className="w-[120px]">
              {t('apps.outputGuardrail')}
            </Label>
          </div>

          <div className="space-y-1">
            <Input
              className="w-120"
              value={botConfig.guardrail_hint}     
              placeholder={t('apps.guardrailHint')}
              onChange={(e) => onConfigChange({ guardrail_hint: e.target.value })}
            />
            <Label htmlFor="guardrail_hint" className="w-[200px]">
              {t('apps.guardrailHintTip')}
            </Label>
          </div>
        </div>
      </div>

      <div className="sticky bottom-0 z-10 -mx-6 px-6 pt-4 flex gap-6 justify-center items-center bg-background border-t">
        <Button
          variant="secondary"
          className="w-20"
          onClick={() => {
            router.push('/apps');
          }}
        >
          {t('common.cancel')}
        </Button>

        <Button
          className="w-40"
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? t('common.saving') : (isCreate ? t('apps.createApp') : t('apps.saveApp'))}
        </Button>
      </div>
    </div>
  );
};
