'use client';

import React, { useState, useEffect } from 'react';
import { TrashIcon, Edit, AlertCircleIcon, Plus, Bot, MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { LLMModelDialog } from '@/app/config/model/llm/modelDialog';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export interface LlmConfig {
  id: string;
  model_id: string;
  source: string;
  model: string;
  api_key: string;
  base_url: string;
  max_context: number;
  enabled: boolean;
  vision_support: boolean;
  enable_thinking: boolean;
  temperature: number;
}

const newllmconfig: LlmConfig = {
  id: '',
  model_id: '',
  source: '',
  model: '',
  base_url: '',
  api_key: '',
  vision_support: false,
  max_context: 0,
  enabled: true,
  enable_thinking: false,
  temperature: 0.1,
};
export default function LlmConfigPage() {
  const { t } = useI18n();
  const [editLlmConfig, setEditLlmConfig] = useState<LlmConfig>(newllmconfig);
  const [llmconfigs, setLlmConfigs] = useState<LlmConfig[]>([]);
  const [errorMsg, setErrorMsg] = useState('');

  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const res = await tenantFetch(`/api/config/llms?page=1&size=100`);
        if (!res.ok) throw new Error(t('config.model.fetchLlmListFailed'));
        const json_data = await res.json();
        setLlmConfigs(json_data.data.items || []);
      } catch (err: any) {
        // swallow — error reported via errorMsg on mutations
      }
    };
    fetchModelConfigs();
  }, [llmconfigs.length]);

  const handleCreateSuccess = (llmConfig: LlmConfig) => {
    setLlmConfigs((prev) => [...prev, llmConfig]);
    setEditLlmConfig(newllmconfig);
  };

  const handleSaveSuccess = (llmConfig: LlmConfig) => {
    setLlmConfigs((prev) =>
      prev.map((config) => (config.id === llmConfig.id ? llmConfig : config)),
    );
    setEditLlmConfig(newllmconfig);
  };

  const removeModel = async (id: string) => {
    setErrorMsg('');
    try {
      const res = await tenantFetch(`/api/config/llms/${id}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!res.ok) {
        setErrorMsg(t('config.model.deleteModelFailed', { modelType: 'LLM' }));
        return;
      }
      setLlmConfigs((prev) => prev.filter((config) => config.id !== id));
    } catch (err: any) {
      setErrorMsg(t('config.model.deleteFailed'));
    }
  };

  const openCreate = () => {
    setIsCreateOpen(true);
    setEditLlmConfig(newllmconfig);
  };

  return (
    <section id="llm" className="section-panel">
      {/* Section header */}
      <div className="section-header">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-primary" />
          <h2 className="section-title">{t('config.model.tabLlm') || 'LLM'}</h2>
          <span className="text-xs text-muted-foreground">· {llmconfigs.length}</span>
        </div>
        <Button size="sm" variant="outline" onClick={openCreate}>
          <Plus className="w-4 h-4" />
          Add
        </Button>
      </div>

      <LLMModelDialog
        isAdd={isCreateOpen ? true : false}
        isOpen={isEditOpen || isCreateOpen}
        setIsOpen={(open) => {
          if (!open) {
            setEditLlmConfig(newllmconfig);
          }
          setIsEditOpen(open);
          setIsCreateOpen(open);
        }}
        llmConfig={editLlmConfig || newllmconfig}
        onSaveSuccess={(llm) => {
          if (isCreateOpen) handleCreateSuccess(llm);
          else handleSaveSuccess(llm);
        }}
      />

      {llmconfigs.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-3">
          {llmconfigs.map((llm) => (
            <div key={llm.id} className="model-card group">
              {/* Top-right corner: type badge + more menu */}
              <div className="absolute top-3 right-3 flex items-center gap-2">
                <span className="type-corner-badge type-llm">LLM</span>
                <DropdownMenu modal={false}>
                  <DropdownMenuTrigger asChild>
                    <button type="button" className="icon-action-btn" aria-label="More">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="menu-compact">
                    <DropdownMenuItem
                      onSelect={() => {
                        setEditLlmConfig(llm);
                        setIsEditOpen(true);
                      }}
                    >
                      <Edit />
                      {t('common.edit') || 'Edit'}
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      className="text-destructive focus:text-destructive"
                      onSelect={() => removeModel(llm.id)}
                    >
                      <TrashIcon />
                      {t('common.delete') || 'Delete'}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <div className="flex items-start gap-3 pr-20">
                <div className="model-icon type-llm shrink-0">{(llm.model_id || 'M').charAt(0)}</div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm truncate" title={llm.model_id}>
                    {llm.model_id}
                  </div>
                  <div className="text-xs text-muted-foreground truncate">{llm.model}</div>
                </div>
              </div>

              <div className="flex items-center gap-1.5 flex-wrap">
                <Badge className="badge-tech text-[10.5px] py-0">{llm.source}</Badge>
                {llm.vision_support && (
                  <Badge className="badge-warning text-[10.5px] py-0">
                    {t('config.model.visionModel')}
                  </Badge>
                )}
                {llm.enable_thinking && (
                  <Badge className="badge-warning text-[10.5px] py-0">
                    {t('config.model.thinkingModel')}
                  </Badge>
                )}
              </div>

              {llm.base_url && (
                <div className="endpoint-text" title={llm.base_url}>
                  {llm.base_url}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <Bot className="w-10 h-10 text-muted-foreground/40 mb-2" />
          <p className="text-sm text-muted-foreground">
            暂无模型，点击右上角按钮添加
          </p>
        </div>
      )}

      {errorMsg !== '' && (
        <div className="block w-full mt-3">
          <Alert variant="destructive">
            <AlertCircleIcon />
            <AlertDescription>
              <p>{errorMsg}</p>
            </AlertDescription>
          </Alert>
        </div>
      )}
    </section>
  );
}
