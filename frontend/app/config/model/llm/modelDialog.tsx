// LLMModelDialog.tsx
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
import { Switch } from '@/components/ui/switch';
import { Alert, AlertTitle } from '@/components/ui/alert';
import { AlertCircleIcon } from 'lucide-react';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

// 定义组件 props
interface LLMModelDialogProps {
  isAdd: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  llmConfig: LlmConfig;
  onSaveSuccess: (llm: LlmConfig) => void;
}

// 模型数据类型
interface LlmConfig {
  id: string;
  model_id: string;
  source: string;
  model: string;
  api_key: string;
  base_url: string;
  max_context: number;
  enabled: boolean;
  vision_support: boolean;
  enable_thinking: boolean; // 是否支持思考模式
}

export const LLMModelDialog: FC<LLMModelDialogProps> = ({
  isAdd,
  isOpen,
  setIsOpen,
  llmConfig,
  onSaveSuccess,
}) => {
  const { t } = useI18n();
  const [llm, setLlm] = useState<LlmConfig>(llmConfig);
  const [error, setError] = useState<string | null>(null);
  const [saveErrorMsg, setSaveErrorMsg] = useState('');
  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    setLlm(llmConfig);
  }, [isAdd, llmConfig]);

  useEffect(() => {
    setSaveErrorMsg('');
  }, [llm]);

  const handleSubmit = async () => {
    setSaveErrorMsg('');
    if (
      !llm.model ||
      (isAdd && !llm.api_key) ||
      !llm.base_url ||
      !llm.model_id
    ) {
      setSaveErrorMsg(t('config.model.fillCompleteInfo'));
      return;
    }
    const submit_url = isAdd ? `/api/config/llms` : `/api/config/llms/${llm.id}`;
    const updateMethod = isAdd ? 'POST' : 'PUT';
    if (llm.api_key === '******') llm.api_key = '';
    console.log('updateMethod', isAdd, updateMethod, submit_url, llm);
    try {
      const res = await tenantFetch(submit_url, {
        method: updateMethod,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(llm),
      });

      if (!res.ok) {
        setSaveErrorMsg(t('config.model.requestFailedCheckInfo', { method: updateMethod }));
        return;
      }
      const jsondata = await res.json();
      onSaveSuccess(jsondata.data as LlmConfig); // 触发回调
      setIsOpen(false);
    } catch (err: any) {
      setSaveErrorMsg(t('config.model.requestFailed', { method: updateMethod }));
    }
  };

  // 对话框关闭时重置表单
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
            <Label htmlFor="model_id" className="text-right">
              {t('config.model.modelId')}
              <span className="text-destructive">*</span>
            </Label>
            <Input
              id="model_id"
              placeholder={t('config.model.modelIdPlaceholder')}
              value={llm?.model_id ?? ''}
              onChange={(e) =>
                setLlm((prev) => ({ ...prev, model_id: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
        <div>
          <div className="grid grid-cols-4 items-center gap-4 py-2">
            <Label htmlFor="base_url" className="text-right">
              {t('config.model.endpointUrl')}
              <span className="text-destructive">*</span>
            </Label>
            <div className="col-span-3">
              <input
                id="base_url"
                list="base_url_options"
                placeholder={t('config.model.baseUrlPlaceholder')}
                value={llm?.base_url ?? ''}
                onChange={(e) =>
                  setLlm((prev) => ({ ...prev, base_url: e.target.value }))
                }
                className="w-full border border-gray-300 rounded-md p-2 text-sm"
              />
              <datalist id="base_url_options">
                <option value="https://api.openai.com/v1">OpenAI</option>
                <option value="https://dashscope.aliyuncs.com/compatible-mode/v1">
                  {t('config.model.qwenModel')}
                </option>
                {/* 添加更多预设选项 */}
              </datalist>
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
                value={llm?.api_key ?? ''}
                onChange={(e) =>
                  setLlm((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            ) : (
              <Input
                id="api_key"
                type="password"
                placeholder={t('config.model.apiKeyPlaceholder')}
                value={llm?.api_key || '******'}
                onChange={(e) =>
                  setLlm((prev) => ({ ...prev, api_key: e.target.value }))
                }
                className="col-span-3"
              />
            )}
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="model" className="text-right">
              {t('config.model.modelName')}
              <span className="text-destructive">*</span>
            </Label>
            <Input
              id="model"
              placeholder={t('config.model.modelNamePlaceholder')}
              value={llm?.model ?? ''}
              onChange={(e) =>
                setLlm((prev) => ({ ...prev, model: e.target.value }))
              }
              className="col-span-3"
            />
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="vision_support" className="text-right">
              {t('config.model.visionModel')}
            </Label>
            <Switch
              id="vision_support"
              checked={llm?.vision_support ?? false}
              onCheckedChange={(checked) =>
                setLlm((prev) => ({ ...prev, vision_support: checked }))
              }
            />
          </div>
        </div>
        <div className="grid gap-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="enable_thinking" className="text-right">
              {t('config.model.thinkingModel')}
            </Label>
            <Switch
              id="enable_thinking"
              checked={llm?.enable_thinking ?? false}
              onCheckedChange={(checked) =>
                setLlm((prev) => ({ ...prev, enable_thinking: checked }))
              }
            />
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
