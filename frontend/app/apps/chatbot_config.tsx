'use client';
import React, { useState, useEffect, FC } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  ChevronDownIcon,
  Info,
  ShieldCheck,
  Sparkles,
  Database,
  Boxes,
  Pencil,
  Image,
} from 'lucide-react';
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
  vision_model_id?: string | null;
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

// Compact section: icon + title (description inline as subtitle)
export const Section: FC<{
  icon: React.ReactNode;
  title: string;
  description?: string;
  rightSlot?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}> = ({ icon, title, description, rightSlot, children, className = '' }) => (
  <section className={`py-3.5 border-b border-border last:border-b-0 ${className}`}>
    <div className="flex items-center gap-2 mb-2.5">
      <div className="flex items-center justify-center w-6 h-6 rounded-md bg-primary/10 text-primary shrink-0">
        {icon}
      </div>
      <h3 className="text-sm font-semibold leading-tight">{title}</h3>
      {description && (
        <span className="text-xs text-muted-foreground truncate">· {description}</span>
      )}
      {rightSlot && <div className="ml-auto shrink-0">{rightSlot}</div>}
    </div>
    <div className="space-y-3 pl-8">{children}</div>
  </section>
);

// Toggle row with title + hint, switch on the right
const ToggleRow: FC<{
  id: string;
  title: string;
  hint?: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}> = ({ id, title, hint, checked, onCheckedChange }) => (
  <div className="flex items-center justify-between rounded-md px-3 py-1.5 bg-muted/40 hover:bg-muted/70 transition-colors">
    <div className="flex flex-col gap-0.5 min-w-0">
      <span className="text-[13px] font-medium truncate">{title}</span>
      {hint && <span className="text-[11px] text-muted-foreground truncate">{hint}</span>}
    </div>
    <Switch id={id} checked={checked} onCheckedChange={onCheckedChange} />
  </div>
);

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
  const [systemPrompt, setSystemPrompt] = useState('');
  const [defaultPrompts, setDefaultPrompts] = useState({
    react: REACT_PROMPT,
  });

  const router = useRouter();
  const visionLlms = llms.filter((llm) => llm.vision_support);

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

  return (
    <div className="max-w-5xl mx-auto px-6">
      {/* Section 1: 基本信息 */}
      <Section
        icon={<Info className="w-4 h-4" />}
        title={t('apps.sectionBasicInfo')}
        description={t('apps.sectionBasicInfoDesc')}
      >
        {/* 第一行: 描述 (创建时左侧加 App ID) */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-start">
          {isCreate && (
            <div className="space-y-1.5 md:col-span-4">
              <Label htmlFor="appid" className="text-xs font-medium">
                App ID <span className="text-destructive">*</span>
              </Label>
              <Input
                id="appid"
                value={botConfig.app_id || ''}
                onChange={(e) => onConfigChange({ app_id: e.target.value })}
                placeholder={t('apps.appIdPlaceholder')}
                required
                className="h-9"
              />
            </div>
          )}
          <div className={`space-y-1.5 ${isCreate ? 'md:col-span-8' : 'md:col-span-12'}`}>
            <Label htmlFor="description" className="text-xs font-medium">
              {t('apps.descriptionLabel')}
            </Label>
            <Input
              id="description"
              value={botConfig.description || ''}
              onChange={(e) => onConfigChange({ description: e.target.value })}
              placeholder={t('apps.descriptionPlaceholder')}
              className="h-9"
            />
          </div>
        </div>

        {/* 第二行: 基模型 + 编辑提示词 */}
        <div className="space-y-1.5">
          <Label htmlFor="basemodel" className="text-xs font-medium">
            {t('apps.baseModel')} <span className="text-destructive">*</span>
          </Label>
          <div className="flex gap-2">
            <div className="flex-1">
              {llms.length > 0 ? (
                <Select
                  value={botConfig.model_id || ''}
                  onValueChange={(value) => onConfigChange({ model_id: value })}
                >
                  <SelectTrigger className="w-full h-9">
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
                <Button
                  variant="outline"
                  className="w-full justify-start text-muted-foreground font-normal h-9"
                  onClick={() => router.push('/config/model/llm')}
                >
                  {t('apps.noModelConfigured')} — {t('apps.addModel')}
                </Button>
              )}
            </div>
            <Dialog open={openPrompt} onOpenChange={setOpenPrompt}>
              <DialogTrigger asChild>
                <Button
                  size="sm"
                  className="shrink-0 h-9 bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 shadow-none"
                >
                  <Pencil className="w-3.5 h-3.5 mr-1.5" />
                  {t('apps.editPrompts')}
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-2xl lg:max-w-4xl max-h-[90vh] flex flex-col gap-4">
                <DialogHeader className="space-y-1">
                  <DialogTitle className="flex items-center gap-2">
                    <Pencil className="w-4 h-4 text-primary" />
                    {t('apps.editPrompts')}
                  </DialogTitle>
                  <DialogDescription className="text-xs">
                    {t('apps.editPromptDesc')}
                  </DialogDescription>
                </DialogHeader>

                <div className="flex-1 overflow-y-auto -mx-1 px-1">
                  <ResettableTextarea
                    value={systemPrompt}
                    onReset={() => setSystemPrompt(defaultPrompts.react)}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    defaultValue={defaultPrompts.react}
                    resetLabel={t('common.reset') || '重置默认'}
                  />
                </div>

                <DialogFooter className="gap-2 sm:gap-0 border-t border-border pt-3">
                  <div className="flex items-center justify-between w-full">
                    <p
                      className="text-xs text-muted-foreground flex items-center gap-1.5"
                      suppressHydrationWarning
                    >
                      <Info className="w-3.5 h-3.5" />
                      {t('apps.promptSaveReminder')}
                    </p>
                    <div className="flex gap-2">
                      <DialogClose asChild>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            setSystemPrompt(botConfig.prompts?.react || defaultPrompts.react);
                          }}
                        >
                          {t('common.cancel')}
                        </Button>
                      </DialogClose>
                      <Button
                        type="button"
                        size="sm"
                        onClick={async () => {
                          onConfigChange({
                            prompts: {
                              react: systemPrompt,
                            },
                          });
                          setOpenPrompt(false);
                        }}
                      >
                        {t('common.save')}
                      </Button>
                    </div>
                  </div>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="visionmodel" className="text-xs font-medium">
            {t('apps.visionModel')}
          </Label>
          {visionLlms.length > 0 ? (
            <Select
              value={botConfig.vision_model_id || 'DISABLED'}
              onValueChange={(value) =>
                onConfigChange({ vision_model_id: value === 'DISABLED' ? null : value })
              }
            >
              <SelectTrigger className="w-full h-9">
                <SelectValue placeholder={t('apps.selectVisionModel')} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="DISABLED">{t('apps.disableVisionModel')}</SelectItem>
                {visionLlms.map((llm) => (
                  <SelectItem key={llm.id} value={llm.model_id}>
                    {llm.model || llm.model_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <Button
              variant="outline"
              className="w-full justify-start text-muted-foreground font-normal h-9"
              onClick={() => router.push('/config/model/llm')}
            >
              {t('apps.noVisionModelConfigured')} — {t('apps.addModel')}
            </Button>
          )}
          <p className="text-[11px] text-muted-foreground flex items-center gap-1.5">
            <Image className="w-3.5 h-3.5" />
            {t('apps.visionModelHint')}
          </p>
        </div>
      </Section>

      {/* Section 2: 功能开关 */}
      <Section
        icon={<Sparkles className="w-4 h-4" />}
        title={t('apps.sectionFeatures')}
        description={t('apps.sectionFeaturesDesc')}
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <ToggleRow
            id="enable_search"
            title={t('apps.enableSearch')}
            checked={botConfig.enable_search || false}
            onCheckedChange={(checked) => onConfigChange({ enable_search: checked })}
          />
          <ToggleRow
            id="enable_chatdb"
            title={t('apps.enableChatDb')}
            checked={botConfig.enable_chatdb || false}
            onCheckedChange={(checked) => onConfigChange({ enable_chatdb: checked })}
          />
        </div>
      </Section>

      {/* Section 3: 知识库 + MCP 工具 (并列两列) */}
      <section className="py-3.5 border-b border-border">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3">
          {/* 知识库 */}
          <div>
            <div className="flex items-center gap-2 mb-2.5">
              <div className="flex items-center justify-center w-6 h-6 rounded-md bg-primary/10 text-primary shrink-0">
                <Database className="w-4 h-4" />
              </div>
              <h3 className="text-sm font-semibold leading-tight">{t('apps.knowledgebaseSelection')}</h3>
            </div>
            <div className="space-y-2 pl-8">
              <div className="flex items-center gap-2 flex-wrap">
                {kbs.length > 0 ? (
                  <DropdownMenu modal={true}>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm" className="shrink-0 h-8">
                        {t('apps.selectedKbNum', { num: botConfig?.kb_ids?.length || 0 })}
                        <ChevronDownIcon className="ml-1 w-4 h-4" />
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
                  <p className="text-xs text-muted-foreground">{t('apps.noKbConfigured')}</p>
                )}
                {selectedKbNames.length > 0 && (
                  <div className="flex gap-1 items-center flex-wrap">
                    {selectedKbNames.map((name) => (
                      <Badge variant="secondary" className="h-6 text-xs" key={name}>
                        {name}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
              <ToggleRow
                id="enable_auto_metadata_filter"
                title={t('apps.enableAutoMetadataFilter')}
                checked={botConfig.enable_auto_metadata_filter || false}
                onCheckedChange={(checked) => onConfigChange({ enable_auto_metadata_filter: checked })}
              />
            </div>
          </div>

          {/* MCP工具 */}
          <div>
            <div className="flex items-center gap-2 mb-2.5">
              <div className="flex items-center justify-center w-6 h-6 rounded-md bg-primary/10 text-primary shrink-0">
                <Boxes className="w-4 h-4" />
              </div>
              <h3 className="text-sm font-semibold leading-tight">{t('apps.mcpSelection')}</h3>
            </div>
            <div className="space-y-2 pl-8">
              <div className="flex items-center gap-2 flex-wrap">
                {mcps.length > 0 ? (
                  <DropdownMenu modal={true}>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm" className="shrink-0 h-8">
                        {t('apps.selectedMcpNum', { num: botConfig.mcp_ids?.length || 0 })}
                        <ChevronDownIcon className="ml-1 w-4 h-4" />
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
                  <p className="text-xs text-muted-foreground">{t('apps.noMcpConfigured')}</p>
                )}
                {selectedMcpNames.length > 0 && (
                  <div className="flex gap-1 items-center flex-wrap">
                    {selectedMcpNames.map((name) => (
                      <Badge variant="secondary" className="h-6 text-xs" key={name}>
                        {name}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 4: 安全护栏 */}
      <Section
        icon={<ShieldCheck className="w-4 h-4" />}
        title={t('apps.guardrail')}
      >
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-end">
          <div className="md:col-span-3">
            <ToggleRow
              id="enable_input_check"
              title={t('apps.inputGuardrail')}
              checked={botConfig.enable_input_guardrail || false}
              onCheckedChange={(checked) => onConfigChange({ enable_input_guardrail: checked })}
            />
          </div>
          <div className="md:col-span-3">
            <ToggleRow
              id="enable_output_check"
              title={t('apps.outputGuardrail')}
              checked={botConfig.enable_output_guardrail || false}
              onCheckedChange={(checked) => onConfigChange({ enable_output_guardrail: checked })}
            />
          </div>
          <div className="md:col-span-6 space-y-1.5">
            <Label htmlFor="guardrail_hint" className="text-xs font-medium text-muted-foreground">
              {t('apps.guardrailHintTip')}
            </Label>
            <Input
              id="guardrail_hint"
              value={botConfig.guardrail_hint}
              placeholder={t('apps.guardrailHint')}
              onChange={(e) => onConfigChange({ guardrail_hint: e.target.value })}
              className="h-9"
            />
          </div>
        </div>
      </Section>

    </div>
  );
};
