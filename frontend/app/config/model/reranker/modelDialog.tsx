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
  is_multimodal?: boolean;
}

export const RerankerModelDialog: FC<RerankerModelDialogProps> = ({
  isAdd,
  isOpen,
  setIsOpen,
  rerankerConfig,
  onSaveSuccess,
}) => {
  const [reranker, setReranker] = useState<RerankerConfig>(rerankerConfig);
  const [saveErrorMsg, setSaveErrorMsg] = useState('');
  const { tenantFetch } = useTenantFetch();
  const { t } = useI18n();

  // Freeze isAdd while the dialog is closing (prevents title flicker).
  const [displayIsAdd, setDisplayIsAdd] = useState(isAdd);
  useEffect(() => {
    if (isOpen) setDisplayIsAdd(isAdd);
  }, [isOpen, isAdd]);

  useEffect(() => {
    setReranker(rerankerConfig);
  }, [isAdd, rerankerConfig]);

  useEffect(() => {
    setSaveErrorMsg('');
  }, [reranker]);

  const handleSubmit = async () => {
    setSaveErrorMsg('');
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

    const typeMapping: Record<string, string> = {
      OpenAICompatible: 'openai_like',
      DashScope: 'dashscope',
      MultimodalDashScope: 'multimodal_dashscope',
    };
    const backendType = reranker.type
      ? typeMapping[reranker.type] || reranker.type
      : 'openai_like';
    const submitData = {
      ...reranker,
      type: backendType,
      is_multimodal: backendType === 'multimodal_dashscope'
        ? true
        : Boolean(reranker.is_multimodal),
    };

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
      const reverseTypeMapping: Record<string, string> = {
        openai_like: 'OpenAICompatible',
        dashscope: 'DashScope',
        multimodal_dashscope: 'MultimodalDashScope',
      };
      const responseData = {
        ...jsondata.data,
        type: jsondata.data.type
          ? reverseTypeMapping[jsondata.data.type] || jsondata.data.type
          : 'OpenAICompatible',
      };
      onSaveSuccess(responseData as RerankerConfig);
      setIsOpen(false);
    } catch (err: any) {
      setSaveErrorMsg(t('config.model.requestFailed', { method: updateMethod }));
    }
  };

  const handleDialogClose = (open: boolean) => {
    setIsOpen(open);
    if (!open) setSaveErrorMsg('');
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleDialogClose}>
      <DialogContent className="sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <span className="type-corner-badge type-reranker">RNK</span>
            {displayIsAdd ? '添加 Reranker 模型' : '编辑 Reranker 模型'}
          </DialogTitle>
          <DialogDescription>
            填写 Reranker（重排序）模型的配置信息后保存
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
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
                  value={reranker?.model_id ?? ''}
                  onChange={(e) =>
                    setReranker((prev) => ({ ...prev, model_id: e.target.value }))
                  }
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
                  value={reranker?.model_name ?? ''}
                  onChange={(e) =>
                    setReranker((prev) => ({ ...prev, model_name: e.target.value }))
                  }
                />
              </div>

              <div>
                <label htmlFor="type" className="form-label">
                  {t('config.model.modelType') || 'Type'}
                </label>
                <Select
                  value={reranker?.type || 'OpenAICompatible'}
                  onValueChange={(v) =>
                    setReranker((prev) => ({
                      ...prev,
                      type: v,
                      is_multimodal: v === 'MultimodalDashScope',
                    }))
                  }
                >
                  <SelectTrigger id="type" className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="OpenAICompatible">OpenAI Compatible</SelectItem>
                    <SelectItem value="DashScope">DashScope</SelectItem>
                    <SelectItem value="MultimodalDashScope">
                      {t('config.model.multimodalDashscope')}
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          {/* Endpoint section */}
          <div>
            <div className="dialog-section-title">Endpoint</div>
            <div className="space-y-3">
              <div>
                <label htmlFor="base_url" className="form-label">
                  {t('config.model.endpointUrl')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="base_url"
                  placeholder={t('config.model.baseUrlPlaceholder')}
                  value={reranker?.base_url ?? ''}
                  onChange={(e) =>
                    setReranker((prev) => ({ ...prev, base_url: e.target.value }))
                  }
                />
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
                  value={isAdd ? (reranker?.api_key ?? '') : (reranker?.api_key || '******')}
                  onChange={(e) =>
                    setReranker((prev) => ({ ...prev, api_key: e.target.value }))
                  }
                />
              </div>
            </div>
          </div>

          {saveErrorMsg !== '' && (
            <Alert className="bg-destructive/10 dark:bg-destructive/20 border-none">
              <AlertCircleIcon className="h-4 w-4 !text-destructive" />
              <AlertTitle>{saveErrorMsg}</AlertTitle>
            </Alert>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            {t('common.cancel') || 'Cancel'}
          </Button>
          <Button onClick={handleSubmit}>
            {displayIsAdd ? '创建 Reranker 模型' : t('common.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
