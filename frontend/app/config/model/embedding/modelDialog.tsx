// EmbeddingModelDialog.tsx
import { useState, useEffect, FC } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertTitle } from '@/components/ui/alert';
import { AlertCircleIcon } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';
import { cn } from '@/lib/utils';

interface EmbeddingModelDialogProps {
  isAdd: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  embConfig: EmbConfig;
  onSaveSuccess: (emb: EmbConfig) => void;
}

export interface EmbConfig {
  id: string;
  model_id: string;
  model_name: string;
  type: string;
  api_key: string;
  endpoint: string;
  dimension: number | undefined;
  embed_batch_size: number;
  is_ready: boolean;
  is_default: boolean;
  is_multimodal: boolean;
}

export const EmbeddingModelDialog: FC<EmbeddingModelDialogProps> = ({
  isAdd,
  isOpen,
  setIsOpen,
  embConfig,
  onSaveSuccess,
}) => {
  const [emb, setEmb] = useState<EmbConfig>(embConfig);
  const [saveErrorMsg, setSaveErrorMsg] = useState('');
  const { tenantFetch } = useTenantFetch();
  const { t } = useI18n();

  // Freeze isAdd while the dialog is closing (prevents title flicker).
  const [displayIsAdd, setDisplayIsAdd] = useState(isAdd);
  useEffect(() => {
    if (isOpen) setDisplayIsAdd(isAdd);
  }, [isOpen, isAdd]);

  useEffect(() => {
    setEmb(embConfig);
  }, [isAdd, embConfig]);

  useEffect(() => {
    setSaveErrorMsg('');
  }, [emb]);

  const handleSubmit = async () => {
    setSaveErrorMsg('');
    const is_api_model = emb.type != 'local';
    const is_multimodal_type = emb.type === 'multimodal_dashscope';
    if (emb.dimension === 0) emb.dimension = undefined;

    if (
      is_api_model &&
      (!emb.model_id ||
        (isAdd && !emb.api_key) ||
        !emb.endpoint ||
        !emb.model_name ||
        !emb.type ||
        (is_multimodal_type && !emb.dimension))
    ) {
      setSaveErrorMsg(t('config.model.fillCompleteInfo'));
      return;
    } else if (
      !is_api_model &&
      (!emb.model_id || !emb.model_name || !emb.dimension || !emb.type)
    ) {
      setSaveErrorMsg(t('config.model.fillCompleteInfo'));
      return;
    }

    if (is_multimodal_type) {
      emb.is_multimodal = true;
    } else if (emb.is_multimodal) {
      emb.is_multimodal = false;
    }
    const submit_url = isAdd
      ? `/api/config/embeddings`
      : `/api/config/embeddings/${emb.id}`;
    const updateMethod = isAdd ? 'POST' : 'PUT';
    if (emb.api_key === '******') emb.api_key = '';
    try {
      const res = await tenantFetch(submit_url, {
        method: updateMethod,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(emb),
      });

      if (!res.ok) {
        setSaveErrorMsg(t('config.model.requestFailedCheckInfo', { method: updateMethod }));
        return;
      }
      const jsondata = await res.json();
      onSaveSuccess(jsondata.data as EmbConfig);
      setIsOpen(false);
    } catch (err: unknown) {
      setSaveErrorMsg(t('config.model.requestFailed', { method: updateMethod }));
    }
  };

  const handleDialogClose = (open: boolean) => {
    setIsOpen(open);
    if (!open) setSaveErrorMsg('');
  };

  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [pendingState, setPendingState] = useState<boolean | null>(null);

  const isMultimodal = emb?.type === 'multimodal_dashscope';
  const isApiLike = emb?.type === 'openai_like' || isMultimodal;

  return (
    <Dialog open={isOpen} onOpenChange={handleDialogClose}>
      <DialogContent className="sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <span className="type-corner-badge type-embedding">EMB</span>
            {displayIsAdd ? '添加 Embedding 模型' : '编辑 Embedding 模型'}
          </DialogTitle>
          <DialogDescription>
            填写 Embedding（向量）模型的配置信息后保存
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {/* Type selector — segmented */}
          <div>
            <label className="form-label">
              {t('config.model.modelType')}
              <span className="required">*</span>
            </label>
            <div className="grid grid-cols-3 gap-2 p-1 rounded-lg bg-muted">
              <button
                type="button"
                className={cn(
                  'px-3 py-2 text-sm rounded-md transition-all',
                  emb?.type === 'local'
                    ? 'bg-background shadow-sm font-medium text-foreground'
                    : 'text-muted-foreground hover:text-foreground',
                )}
                onClick={() => setEmb({ ...emb, type: 'local', is_multimodal: false })}
              >
                {t('config.model.localHosted')}
              </button>
              <button
                type="button"
                className={cn(
                  'px-3 py-2 text-sm rounded-md transition-all',
                  emb?.type === 'openai_like'
                    ? 'bg-background shadow-sm font-medium text-foreground'
                    : 'text-muted-foreground hover:text-foreground',
                )}
                onClick={() => setEmb({ ...emb, type: 'openai_like', is_multimodal: false })}
              >
                {t('config.model.apiOpenaiLike')}
              </button>
              <button
                type="button"
                className={cn(
                  'px-3 py-2 text-sm rounded-md transition-all',
                  emb?.type === 'multimodal_dashscope'
                    ? 'bg-background shadow-sm font-medium text-foreground'
                    : 'text-muted-foreground hover:text-foreground',
                )}
                onClick={() => {
                  const isCompatibleMode =
                    !emb.endpoint ||
                    emb.endpoint.includes('/compatible-mode');
                  setEmb({
                    ...emb,
                    type: 'multimodal_dashscope',
                    is_multimodal: true,
                    endpoint: isCompatibleMode
                      ? 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding'
                      : emb.endpoint,
                  });
                }}
              >
                {t('config.model.multimodalDashscope')}
              </button>
            </div>
          </div>

          {/* Basic section */}
          <div>
            <div className="dialog-section-title">Basic</div>
            <div className="space-y-3">
              <div>
                <label htmlFor="model_id" className="form-label">
                  {t('config.model.modelId')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="model_id"
                  placeholder={t('config.model.modelIdPlaceholder')}
                  value={emb?.model_id ?? ''}
                  onChange={(e) => setEmb((prev) => ({ ...prev, model_id: e.target.value }))}
                />
              </div>

              <div>
                <label htmlFor="model_name" className="form-label">
                  {t('config.model.modelName')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="model_name"
                  placeholder={t('config.model.modelNamePlaceholder')}
                  value={emb?.model_name ?? ''}
                  onChange={(e) => setEmb((prev) => ({ ...prev, model_name: e.target.value }))}
                />
              </div>
            </div>
          </div>

          {/* Endpoint section (only for API-like) */}
          {isApiLike && (
            <div>
              <div className="dialog-section-title">Endpoint</div>
              <div className="space-y-3">
                <div>
                  <label htmlFor="endpoint" className="form-label">
                    {t('config.model.endpointUrl')}
                    <span className="required">*</span>
                  </label>
                  <Input
                    id="endpoint"
                    list={isMultimodal ? 'mm_endpoint_options' : 'endpoint_options'}
                    placeholder={
                      isMultimodal
                        ? t('config.model.multimodalEndpointPlaceholder')
                        : t('config.model.endpointPlaceholder')
                    }
                    value={emb?.endpoint ?? ''}
                    onChange={(e) => setEmb((prev) => ({ ...prev, endpoint: e.target.value }))}
                  />
                  {isMultimodal ? (
                    <>
                      <p className="text-xs text-muted-foreground mt-1">
                        {t('config.model.multimodalEndpointHint')}
                      </p>
                      <datalist id="mm_endpoint_options">
                        <option value="https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding">
                          {t('config.model.qwenMultimodalEmbedding')}
                        </option>
                      </datalist>
                    </>
                  ) : (
                    <datalist id="endpoint_options">
                      <option value="https://api.openai.com/v1">OpenAI</option>
                      <option value="https://dashscope.aliyuncs.com/compatible-mode/v1">
                        {t('config.model.qwenModel')}
                      </option>
                    </datalist>
                  )}
                </div>

                <div>
                  <label htmlFor="api_key" className="form-label">
                    {t('config.model.apiKey')}
                    <span className="required">*</span>
                  </label>
                  <Input
                    id="api_key"
                    type="password"
                    placeholder={t('config.model.apiKeyPlaceholder')}
                    value={isAdd ? (emb?.api_key ?? '') : (emb?.api_key || '******')}
                    onChange={(e) => setEmb((prev) => ({ ...prev, api_key: e.target.value }))}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Advanced section */}
          <div>
            <div className="dialog-section-title">Advanced</div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="dimension" className="form-label">
                  {t('config.model.vectorDimension')}
                  {(!isApiLike || isMultimodal) && <span className="required">*</span>}
                </label>
                <Input
                  id="dimension"
                  type="number"
                  placeholder={t('config.model.vectorDimensionPlaceholder')}
                  value={emb?.dimension ?? ''}
                  onChange={(e) =>
                    setEmb({ ...emb, dimension: Number(e.target.value) || undefined })
                  }
                />
              </div>

              <div>
                <label htmlFor="embed_batch_size" className="form-label">
                  {t('config.model.vectorBatchSize')}
                </label>
                <Input
                  id="embed_batch_size"
                  type="number"
                  placeholder={t('config.model.vectorBatchSizePlaceholder')}
                  value={emb?.embed_batch_size ?? ''}
                  onChange={(e) =>
                    setEmb({ ...emb, embed_batch_size: Number(e.target.value) })
                  }
                />
              </div>
            </div>

            <div className="form-row-inline mt-3">
              <div className="form-row-inline-label">
                <span className="title">{t('config.model.defaultVectorModel')}</span>
                <span className="hint">作为知识库默认的 Embedding 模型</span>
              </div>
              <Switch
                checked={emb.is_default}
                onCheckedChange={(checked) => {
                  if (emb.is_default === checked) return;
                  setPendingState(checked);
                  setIsDialogOpen(true);
                }}
              />
            </div>
          </div>

          {saveErrorMsg !== '' && (
            <Alert className="bg-destructive/10 dark:bg-destructive/20 border-none">
              <AlertCircleIcon className="h-4 w-4 !text-destructive" />
              <AlertTitle>{saveErrorMsg}</AlertTitle>
            </Alert>
          )}
        </div>

        {/* Confirm default change */}
        <ConfirmDialog
          open={isDialogOpen}
          onOpenChange={setIsDialogOpen}
          variant="warning"
          title={t('config.model.confirmChangeDefaultModel')}
          description={
            pendingState
              ? t('config.model.setAsDefaultWarning')
              : t('config.model.unsetAsDefaultWarning')
          }
          confirmLabel={t('config.model.confirmChange')}
          onConfirm={() => {
            setEmb({ ...emb, is_default: pendingState || false });
            setIsDialogOpen(false);
          }}
        />

        <DialogFooter>
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            {t('common.cancel') || 'Cancel'}
          </Button>
          <Button onClick={handleSubmit}>
            {displayIsAdd ? '创建 Embedding 模型' : t('common.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
