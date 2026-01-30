// RerankerModelDialog.tsx
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
import { Alert, AlertTitle } from '@/components/ui/alert';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { AlertCircleIcon } from 'lucide-react';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

// Component props
interface RerankerModelDialogProps {
  isAdd: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  rerankerConfig: RerankerConfig;
  onSaveSuccess: (reranker: RerankerConfig) => void;
}

interface RerankerConfig {
  id: string;
  model_id: string;
  model_name: string;
  api_key: string;
  base_url: string;
  type?: string;
}

export const RerankerModelDialog: FC<RerankerModelDialogProps> = ({
  isAdd,
  isOpen,
  setIsOpen,
  rerankerConfig,
  onSaveSuccess,
}) => {
  const [reranker, setReranker] = useState<RerankerConfig>(rerankerConfig);
  const [error, setError] = useState<string | null>(null);
  const [saveErrorMsg, setSaveErrorMsg] = useState('');
  const { tenantFetch } = useTenantFetch();
  const { t } = useI18n();
  
  useEffect(() => {
    setReranker(rerankerConfig);
  }, [isAdd, rerankerConfig]);

  useEffect(() => {
    setSaveErrorMsg('');
  }, [reranker]);

  const handleSubmit = async () => {
    setSaveErrorMsg('');
    console.log('reranker', reranker);
    if (
      !reranker.model_id ||
      (isAdd && !reranker.api_key) ||
      !reranker.model_name ||
      !reranker.base_url
    ) {
      setSaveErrorMsg(t('config.model.fillCompleteInfo'));
      return;
    }
    const submit_url = isAdd
      ? `/api/config/rerankers`
      : `/api/config/rerankers/${reranker.id}`;
    const updateMethod = isAdd ? 'POST' : 'PUT';
    if (reranker.api_key === '******') reranker.api_key = '';
    
    // Convert frontend type values to backend format
    const typeMapping: Record<string, string> = {
      'OpenAICompatible': 'openai_like',
      'DashScope': 'dashscope',
    };
    const submitData = {
      ...reranker,
      type: reranker.type ? (typeMapping[reranker.type] || reranker.type) : 'openai_like',
    };
    
    console.log('updateMethod', isAdd, updateMethod, submit_url, submitData);
    try {
      const res = await tenantFetch(submit_url, {
        method: updateMethod,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(submitData),
      });

      if (!res.ok) {
        setSaveErrorMsg(t('config.model.requestFailedCheckInfo', { method: updateMethod }));
        return;
      }
      const jsondata = await res.json();
      // Convert backend type values to frontend format
      const reverseTypeMapping: Record<string, string> = {
        'openai_like': 'OpenAICompatible',
        'dashscope': 'DashScope',
      };
      const responseData = {
        ...jsondata.data,
        type: jsondata.data.type ? (reverseTypeMapping[jsondata.data.type] || jsondata.data.type) : 'OpenAICompatible',
      };
      onSaveSuccess(responseData as RerankerConfig);
      setIsOpen(false);
    } catch (err: any) {
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
            <Label htmlFor="type" className="text-right">
              {t('config.model.modelType')}
            </Label>
            <div className="col-span-3">
              <Select
                value={reranker?.type || 'OpenAICompatible'}
                onValueChange={(value) =>
                  setReranker((prev) => ({ ...prev, type: value }))
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder={t('config.model.selectModelType')} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="OpenAICompatible">{t('config.model.openaiLike')}</SelectItem>
                  <SelectItem value="DashScope">{t('config.model.qwenModel')}</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model_id" className="text-right">
              {t('config.model.modelId')}
              <span className="text-destructive">*</span>
            </Label>
            <Input
              id="model_id"
              placeholder={t('config.model.modelIdPlaceholder')}
              value={reranker?.model_id ?? ''}
              onChange={(e) =>
                setReranker((prev) => ({ ...prev, model_id: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model_name" className="text-right">
              {t('config.model.modelName')}
              <span className="text-destructive">*</span>
            </Label>
            <Input
              id="model_name"
              placeholder={t('config.model.modelNamePlaceholder')}
              value={reranker?.model_name ?? ''}
              onChange={(e) =>
                setReranker((prev) => ({ ...prev, model_name: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
        <div>
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="base_url" className="text-right">
              {t('config.model.baseUrl')}
              <span className="text-destructive">*</span>
            </Label>
            <div className="col-span-3">
              <input
                id="base_url"
                list="base_url_options"
                placeholder={t('config.model.baseUrlPlaceholder')}
                value={reranker?.base_url ?? ''}
                onChange={(e) =>
                  setReranker((prev) => ({ ...prev, base_url: e.target.value }))
                }
                className="w-full border border-gray-300 rounded-md p-2 text-sm"
              />
            </div>
          </div>
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
                value={reranker?.api_key ?? ''}
                onChange={(e) =>
                  setReranker((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            ) : (
              <Input
                id="api_key"
                type="password"
                placeholder={t('config.model.apiKeyPlaceholder')}
                value={reranker?.api_key || '******'}
                onChange={(e) =>
                  setReranker((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            )}
          </div>
        </div>

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
