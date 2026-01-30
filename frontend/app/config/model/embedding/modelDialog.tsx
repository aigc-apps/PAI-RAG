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
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Alert, AlertTitle } from '@/components/ui/alert';
import { AlertCircleIcon } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

// Component props
interface EmbeddingModelDialogProps {
  isAdd: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  embConfig: EmbConfig;
  onSaveSuccess: (emb: EmbConfig) => void;
}

// Model data type
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
}

export const EmbeddingModelDialog: FC<EmbeddingModelDialogProps> = ({
  isAdd,
  isOpen,
  setIsOpen,
  embConfig,
  onSaveSuccess,
}) => {
  const [emb, setEmb] = useState<EmbConfig>(embConfig);
  const [error, setError] = useState<string | null>(null);
  const [saveErrorMsg, setSaveErrorMsg] = useState('');
  const { tenantFetch } = useTenantFetch();
  const { t } = useI18n();

  useEffect(() => {
    setEmb(embConfig);
  }, [isAdd, embConfig]);

  useEffect(() => {
    setSaveErrorMsg('');
  }, [emb]);

  const handleSubmit = async () => {
    setSaveErrorMsg('');
    const is_api_model = emb.type != 'local';
    if (emb.dimension === 0) emb.dimension = undefined;

    console.log('handleSubmit', emb, is_api_model);

    if (
      is_api_model &&
      (!emb.model_id ||
        (isAdd && !emb.api_key) ||
        !emb.endpoint ||
        !emb.model_name ||
        !emb.type)
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
    const submit_url = isAdd
      ? `/api/config/embeddings`
      : `/api/config/embeddings/${emb.id}`;
    const updateMethod = isAdd ? 'POST' : 'PUT';
    if (emb.api_key === '******') emb.api_key = '';
    console.log('updateMethod', isAdd, updateMethod, submit_url, emb);
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

  // Reset form when dialog closes
  const handleDialogClose = (open: boolean) => {
    setIsOpen(open);
    if (!open) {
      setError(null);
      setSaveErrorMsg('');
    }
  };

  // Manage confirmation dialog open state
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  // Temporarily store user's target state
  const [pendingState, setPendingState] = useState<boolean | null>(null);

  return (
    <Dialog open={isOpen} onOpenChange={handleDialogClose}>
      <DialogContent className="sm:max-w-[700px]">
        {error && <div className="text-red-500 mb-4">{error}</div>}
        <DialogHeader>
          <DialogTitle>{isAdd ? t('config.model.addModel') : t('config.model.editModel')}</DialogTitle>
          <DialogDescription>{t('config.model.fillModelConfig')}</DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-2">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model_id" className="text-right">
              {t('config.model.modelId')}
              <span className="text-destructive">*</span>
            </Label>
            <Input
              id="model_id"
              placeholder={t('config.model.modelIdPlaceholder')}
              value={emb?.model_id ?? ''}
              onChange={(e) =>
                setEmb((prev) => ({ ...prev, model_id: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
        {emb?.type === 'openai_like' && (
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="endpoint" className="text-right">
              {t('config.model.endpointUrl')}
              <span className="text-destructive">*</span>
            </Label>
            <div className="col-span-3">
              <input
                id="endpoint"
                list="endpoint_options"
                placeholder={t('config.model.endpointPlaceholder')}
                value={emb?.endpoint ?? ''}
                onChange={(e) =>
                  setEmb((prev) => ({ ...prev, endpoint: e.target.value }))
                }
                className="w-full border border-gray-300 rounded-md p-2 text-sm"
              />
              <datalist id="endpoint_options">
                <option value="https://api.openai.com/v1">OpenAI</option>
                <option value="https://dashscope.aliyuncs.com/compatible-mode/v1">
                  {t('config.model.qwenModel')}
                </option>
              </datalist>
            </div>
          </div>
        )}
        {emb?.type === 'openai_like' && (
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="api_key" className="text-right">
              {t('config.model.apiKey')}
              <span className="text-destructive">*</span>
            </Label>
            {isAdd ? (
              <Input
                id="api_key"
                type="password"
                placeholder={t('config.model.apiKeyPlaceholder')}
                value={emb?.api_key ?? ''}
                onChange={(e) =>
                  setEmb((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            ) : (
              <Input
                id="api_key"
                type="password"
                placeholder={t('config.model.apiKeyPlaceholder')}
                value={emb?.api_key || '******'}
                onChange={(e) =>
                  setEmb((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            )}
          </div>
        )}
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model_name" className="text-right">
              {t('config.model.modelName')}
              <span className="text-destructive">*</span>
            </Label>
            <Input
              id="model_name"
              placeholder={t('config.model.modelNamePlaceholder')}
              value={emb?.model_name ?? ''}
              onChange={(e) =>
                setEmb((prev) => ({ ...prev, model_name: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model_type" className="text-right">
              {t('config.model.modelType')}
              <span className="text-destructive">*</span>
            </Label>
            <RadioGroup
              className="flex flex-row gap-6 col-span-3"
              value={emb?.type}
              onValueChange={(value) => setEmb({ ...emb, type: value })}
            >
              <div className="flex items-center gap-3">
                <RadioGroupItem value="local" />
                <Label>{t('config.model.localHosted')}</Label>
              </div>
              <div className="flex items-center gap-3">
                <RadioGroupItem value="openai_like" />
                <Label>{t('config.model.apiOpenaiLike')}</Label>
              </div>
            </RadioGroup>
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="dimension" className="text-right">
              {t('config.model.vectorDimension')}
            </Label>
            <div className="col-span-3">
              <Input
                id="dimension"
                type="number"
                placeholder={t('config.model.vectorDimensionPlaceholder')}
                defaultValue={emb?.dimension}
                onChange={(e) =>
                  setEmb({
                    ...emb,
                    dimension: Number(e.target.value),
                  })
                }
              />
            </div>
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="embed_batch_size" className="text-right">
              {t('config.model.vectorBatchSize')}
            </Label>
            <Input
              id="embed_batch_size"
              type="number"
              placeholder={t('config.model.vectorBatchSizePlaceholder')}
              defaultValue={emb?.embed_batch_size || 'null'}
              onChange={(e) =>
                setEmb({
                  ...emb,
                  embed_batch_size: Number(e.target.value),
                })
              }
              className="col-span-3"
            />
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="embed_batch_size" className="text-right">
              {t('config.model.defaultVectorModel')}
            </Label>
            <Switch
              checked={emb.is_default}
              className="justify-start rounded-full transition-color"
              onCheckedChange={(checked) => {
                const targetState = checked;
                if (emb.is_default === targetState) return;
                setPendingState(targetState);
                setIsDialogOpen(true);
              }}
            />
          </div>
        </div>
        {/* 确认对话框 */}
        <AlertDialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{t('config.model.confirmChangeDefaultModel')}</AlertDialogTitle>
              <AlertDialogDescription>
                {pendingState
                  ? t('config.model.setAsDefaultWarning')
                  : t('config.model.unsetAsDefaultWarning')}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => {
                  setEmb({
                    ...emb,
                    is_default: pendingState || false,
                  });
                  setIsDialogOpen(false);
                }}
              >
                {t('config.model.confirmChange')}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <DialogFooter className="flex flex-col gap-4">
          {saveErrorMsg !== '' && (
            <Alert className="bg-destructive/10 dark:bg-destructive/20 border-none">
              <AlertCircleIcon className="h-4 w-4 !text-destructive" />
              <AlertTitle>{saveErrorMsg}</AlertTitle>
            </Alert>
          )}
          <Button onClick={handleSubmit}>{isAdd ? t('config.model.create') : t('common.save')}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
