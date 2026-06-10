'use client';

import React, { useState, useEffect } from 'react';
import { TrashIcon, Edit, AlertCircleIcon, Plus, Sparkles, MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { RerankerModelDialog } from '@/app/config/model/reranker/modelDialog';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

interface RerankerConfig {
  id: string;
  model_id: string;
  model_name: string;
  api_key: string;
  base_url: string;
  type?: string;
  is_multimodal?: boolean;
}

const newrerankerconfig: RerankerConfig = {
  id: '',
  model_id: '',
  model_name: '',
  api_key: '',
  base_url: '',
  type: 'OpenAICompatible',
  is_multimodal: false,
};
export default function RerankerConfigPage() {
  const { t } = useI18n();
  const [editRerankerConfig, setEditRerankerConfig] =
    useState<RerankerConfig>(newrerankerconfig);
  const [rerankerconfigs, setRerankerConfigs] = useState<RerankerConfig[]>([]);
  const [errorMsg, setErrorMsg] = useState('');

  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);

  const { tenantFetch } = useTenantFetch();
  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const res = await tenantFetch(`/api/config/rerankers?page=1&size=100`);
        if (!res.ok) throw new Error(t('config.model.fetchRerankerListFailed'));
        const json_data = await res.json();
        const reverseTypeMapping: Record<string, string> = {
          'openai_like': 'OpenAICompatible',
          'dashscope': 'DashScope',
          'multimodal_dashscope': 'MultimodalDashScope',
        };
        const data = (json_data.data.items || []).map((item: RerankerConfig) => ({
          ...item,
          type: item.type ? (reverseTypeMapping[item.type] || item.type) : 'OpenAICompatible',
        }));
        setRerankerConfigs(data);
      } catch (err: any) {
        // swallow
      }
    };
    fetchModelConfigs();
  }, [rerankerconfigs.length]);

  const handleCreateSuccess = (config: RerankerConfig) => {
    setRerankerConfigs((prev) => [...prev, config]);
    setEditRerankerConfig(newrerankerconfig);
  };

  const handleSaveSuccess = (config: RerankerConfig) => {
    setRerankerConfigs((prev) => prev.map((c) => (c.id === config.id ? config : c)));
    setEditRerankerConfig(newrerankerconfig);
  };

  const removeModel = async (id: string) => {
    setErrorMsg('');
    try {
      const res = await tenantFetch(`/api/config/rerankers/${id}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!res.ok) {
        setErrorMsg(t('config.model.deleteModelFailed', { modelType: 'Reranker' }));
        return;
      }
      setRerankerConfigs((prev) => prev.filter((c) => c.id !== id));
    } catch (err: any) {
      setErrorMsg(t('config.model.deleteFailed'));
    }
  };

  const openCreate = () => {
    setIsCreateOpen(true);
    setEditRerankerConfig(newrerankerconfig);
  };

  return (
    <section id="reranker" className="section-panel">
      {/* Section header */}
      <div className="section-header">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          <h2 className="section-title">{t('config.model.tabReranker') || 'Reranker'}</h2>
          <span className="text-xs text-muted-foreground">· {rerankerconfigs.length}</span>
        </div>
        <Button size="sm" variant="outline" onClick={openCreate}>
          <Plus className="w-4 h-4" />
          Add
        </Button>
      </div>

      <RerankerModelDialog
        isAdd={isCreateOpen ? true : false}
        isOpen={isEditOpen || isCreateOpen}
        setIsOpen={(open: boolean) => {
          if (!open) {
            setEditRerankerConfig(newrerankerconfig);
          }
          setIsEditOpen(open);
          setIsCreateOpen(open);
        }}
        rerankerConfig={editRerankerConfig || newrerankerconfig}
        onSaveSuccess={(reranker: RerankerConfig) => {
          if (isCreateOpen) handleCreateSuccess(reranker);
          else handleSaveSuccess(reranker);
        }}
      />

      {rerankerconfigs.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-3">
          {rerankerconfigs.map((reranker) => (
            <div key={reranker.id} className="model-card group">
              {/* Top-right corner: type badge + more menu */}
              <div className="absolute top-3 right-3 flex items-center gap-2">
                <span className="type-corner-badge type-reranker">RNK</span>
                <DropdownMenu modal={false}>
                  <DropdownMenuTrigger asChild>
                    <button type="button" className="icon-action-btn" aria-label="More">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="menu-compact">
                    <DropdownMenuItem
                      onSelect={(e) => {
                        e.preventDefault();
                        setTimeout(() => {
                          setEditRerankerConfig(reranker);
                          setIsEditOpen(true);
                        }, 0);
                      }}
                    >
                      <Edit />
                      {t('common.edit') || 'Edit'}
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      className="text-destructive focus:text-destructive"
                      onSelect={(e) => {
                        e.preventDefault();
                        setTimeout(() => removeModel(reranker.id), 0);
                      }}
                    >
                      <TrashIcon />
                      {t('common.delete') || 'Delete'}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <div className="flex items-start gap-3 pr-20">
                <div className="model-icon type-reranker shrink-0">{(reranker.model_name || 'R').charAt(0)}</div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm truncate" title={reranker.model_id}>
                    {reranker.model_id}
                  </div>
                  <div className="text-xs text-muted-foreground truncate">
                    {reranker.model_name}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-1.5 flex-wrap">
                {reranker.type && (
                  <Badge className="badge-tech text-[10.5px] py-0">{reranker.type}</Badge>
                )}
              </div>

              {reranker.base_url && (
                <div className="endpoint-text" title={reranker.base_url}>
                  {reranker.base_url}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <Sparkles className="w-10 h-10 text-muted-foreground/40 mb-2" />
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
