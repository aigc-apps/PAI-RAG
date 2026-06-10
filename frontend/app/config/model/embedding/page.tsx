'use client';

import React, { useState, useEffect } from 'react';
import {
  TrashIcon,
  Edit,
  AlertCircleIcon,
  CheckCircle,
  Plus,
  Layers,
  MoreHorizontal,
} from 'lucide-react';
import { Spinner } from '@/components/ui/loading';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { EmbeddingModelDialog, EmbConfig } from '@/app/config/model/embedding/modelDialog';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

const newembconfig: EmbConfig = {
  id: '',
  model_id: '',
  model_name: '',
  type: '',
  api_key: '',
  endpoint: '',
  dimension: undefined,
  embed_batch_size: undefined,
  is_ready: false,
  is_default: false,
  is_multimodal: false,
};
export default function EmbConfigPage() {
  const { t } = useI18n();

  const [editEmbConfig, setEditEmbConfig] = useState<EmbConfig>(newembconfig);
  const [embconfigs, setEmbConfigs] = useState<EmbConfig[]>([]);
  const [errorMsg, setErrorMsg] = useState('');

  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const res = await tenantFetch(`/api/config/embeddings?page=1&size=100`);
        if (!res.ok) throw new Error(t('config.model.fetchModelListFailed'));
        const json_data = await res.json();
        setEmbConfigs(json_data.data.items || []);
      } catch (err: any) {
        // swallow
      }
    };
    fetchModelConfigs();
  }, [embconfigs.length, isEditOpen]);

  const handleCreateSuccess = (config: EmbConfig) => {
    setEmbConfigs((prev) => [...prev, config]);
    setEditEmbConfig(newembconfig);
  };

  const handleSaveSuccess = (config: EmbConfig) => {
    setEmbConfigs((prev) => prev.map((c) => (c.id === config.id ? config : c)));
    setEditEmbConfig(newembconfig);
  };

  const removeModel = async (id: string) => {
    setErrorMsg('');
    try {
      const res = await tenantFetch(`/api/config/embeddings/${id}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!res.ok) {
        setErrorMsg(t('config.model.deleteModelFailed', { modelType: 'Embedding' }));
        return;
      }
      setEmbConfigs((prev) => prev.filter((config) => config.id !== id));
    } catch (err: any) {
      setErrorMsg(t('config.model.deleteFailed'));
    }
  };

  const openCreate = () => {
    setIsCreateOpen(true);
    setEditEmbConfig(newembconfig);
  };

  return (
    <section id="embedding" className="section-panel">
      {/* Section header */}
      <div className="section-header">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-primary" />
          <h2 className="section-title">{t('config.model.tabEmbedding') || 'Embedding'}</h2>
          <span className="text-xs text-muted-foreground">· {embconfigs.length}</span>
        </div>
        <Button size="sm" variant="outline" onClick={openCreate}>
          <Plus className="w-4 h-4" />
          Add
        </Button>
      </div>

      <EmbeddingModelDialog
        isAdd={isCreateOpen ? true : false}
        isOpen={isEditOpen || isCreateOpen}
        setIsOpen={(open: boolean) => {
          if (!open) {
            setEditEmbConfig(newembconfig);
          }
          setIsEditOpen(open);
          setIsCreateOpen(open);
        }}
        embConfig={editEmbConfig || newembconfig}
        onSaveSuccess={(emb: EmbConfig) => {
          if (isCreateOpen) handleCreateSuccess(emb);
          else handleSaveSuccess(emb);
        }}
      />

      {embconfigs.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-3">
          {embconfigs.map((emb) => (
            <div key={emb.id} className="model-card group">
              {/* Top-right corner: type badge + default + more menu */}
              <div className="absolute top-3 right-3 flex items-center gap-2">
                {emb.is_default && (
                  <Badge className="badge-danger text-[10.5px] py-0">
                    {t('config.model.default')}
                  </Badge>
                )}
                <span className="type-corner-badge type-embedding">EMB</span>
                <DropdownMenu modal={false}>
                  <DropdownMenuTrigger asChild>
                    <button type="button" className="icon-action-btn" aria-label="More">
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="menu-compact">
                    <DropdownMenuItem
                      onSelect={() => {
                        setEditEmbConfig(emb);
                        setIsEditOpen(true);
                      }}
                    >
                      <Edit />
                      {t('common.edit') || 'Edit'}
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      className="text-destructive focus:text-destructive"
                      onSelect={() => removeModel(emb.id)}
                    >
                      <TrashIcon />
                      {t('common.delete') || 'Delete'}
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <div className="flex items-start gap-3 pr-28">
                <div className="model-icon type-embedding shrink-0">{(emb.model_name || 'E').charAt(0)}</div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm truncate" title={emb.model_name}>
                    {emb.model_name}
                  </div>
                  <div className="model-card-meta" title={emb.model_id}>
                    <span className="meta-key">ID</span>
                    <span className="truncate">{emb.model_id}</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-1.5 flex-wrap">
                <Badge className="badge-tech text-[10.5px] py-0">{emb.type}</Badge>
                {emb.type === 'local' ? (
                  <Badge
                    className={
                      (emb.is_ready ? 'badge-success' : 'badge-neutral') +
                      ' text-[10.5px] py-0'
                    }
                  >
                    {emb.is_ready ? (
                      <span className="inline-flex items-center gap-1">
                        {t('config.model.available')}
                        <CheckCircle className="h-3 w-3" />
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1">
                        {t('config.model.downloading')}
                        <Spinner size="sm" />
                      </span>
                    )}
                  </Badge>
                ) : (
                  <Badge className="badge-success text-[10.5px] py-0">
                    <span className="inline-flex items-center gap-1">
                      {t('config.model.available')}
                      <CheckCircle className="h-3 w-3" />
                    </span>
                  </Badge>
                )}
              </div>

              {emb.endpoint && (
                <div className="endpoint-text" title={emb.endpoint}>
                  {emb.endpoint}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <Layers className="w-10 h-10 text-muted-foreground/40 mb-2" />
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
