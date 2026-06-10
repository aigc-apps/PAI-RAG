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
import { Switch } from '@/components/ui/switch';
import { Alert, AlertTitle } from '@/components/ui/alert';
import { AlertCircleIcon } from 'lucide-react';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

interface LLMModelDialogProps {
  isAdd: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  llmConfig: LlmConfig;
  onSaveSuccess: (llm: LlmConfig) => void;
}

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
  enable_thinking: boolean;
  temperature: number;
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
  const [saveErrorMsg, setSaveErrorMsg] = useState('');
  const [modelIdEdited, setModelIdEdited] = useState(false);
  const { tenantFetch } = useTenantFetch();

  // Freeze isAdd while the dialog is closing (prevents title flicker
  // between "Add" and "Edit" during close animation).
  const [displayIsAdd, setDisplayIsAdd] = useState(isAdd);
  useEffect(() => {
    if (isOpen) setDisplayIsAdd(isAdd);
  }, [isOpen, isAdd]);

  useEffect(() => {
    setLlm(llmConfig);
    setModelIdEdited(false);
  }, [isAdd, llmConfig]);

  useEffect(() => {
    setSaveErrorMsg('');
  }, [llm]);

  const handleSubmit = async () => {
    setSaveErrorMsg('');
    const submitData = {
      ...llm,
      model_id: llm.model_id || llm.model,
    };
    if (!submitData.model || (isAdd && !submitData.api_key) || !submitData.base_url || !submitData.model_id) {
      setSaveErrorMsg(t('config.model.fillCompleteInfo'));
      return;
    }
    const submit_url = isAdd ? `/api/config/llms` : `/api/config/llms/${llm.id}`;
    const updateMethod = isAdd ? 'POST' : 'PUT';
    if (submitData.api_key === '******') submitData.api_key = '';
    try {
      const res = await tenantFetch(submit_url, {
        method: updateMethod,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(submitData),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => null);
        const detail = errorData?.message || errorData?.detail || errorData?.error;
        setSaveErrorMsg(detail || t('config.model.requestFailedCheckInfo', { method: updateMethod }));
        return;
      }
      const jsondata = await res.json();
      onSaveSuccess(jsondata.data as LlmConfig);
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
      <DialogContent className="sm:max-w-[640px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <span className="type-corner-badge type-llm">LLM</span>
            {displayIsAdd ? '添加 LLM 模型' : '编辑 LLM 模型'}
          </DialogTitle>
          <DialogDescription>
            填写服务商调用参数即可，PAI-RAG 模型标识会默认跟随模型名称
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="model-field-guide">
            <div>
              <span className="guide-label">服务商需要</span>
              <span className="guide-value">base_url / api_key / model</span>
            </div>
            <div>
              <span className="guide-label">PAI-RAG 自动生成</span>
              <span className="guide-value">model_id = model</span>
            </div>
          </div>

          {/* Provider section */}
          <div>
            <div className="dialog-section-title">Provider connection</div>
            <div className="space-y-3">
              <div>
                <label htmlFor="base_url" className="form-label">
                  Base URL
                  <span className="required">*</span>
                </label>
                <Input
                  id="base_url"
                  list="base_url_options"
                  placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1"
                  value={llm?.base_url ?? ''}
                  onChange={(e) => setLlm((prev) => ({ ...prev, base_url: e.target.value }))}
                />
                <p className="field-hint">OpenAI 兼容接口地址，通常以 /v1 结尾。</p>
                <datalist id="base_url_options">
                  <option value="https://api.openai.com/v1">OpenAI</option>
                  <option value="https://dashscope.aliyuncs.com/compatible-mode/v1">
                    {t('config.model.qwenModel')}
                  </option>
                </datalist>
              </div>

              <div>
                <label htmlFor="api_key" className="form-label">
                  API Key
                  <span className="required">*</span>
                </label>
                <Input
                  id="api_key"
                  type="password"
                  placeholder="sk-..."
                  value={isAdd ? (llm?.api_key ?? '') : (llm?.api_key || '******')}
                  onChange={(e) => setLlm((prev) => ({ ...prev, api_key: e.target.value }))}
                />
                <p className="field-hint">服务商用于鉴权的密钥，仅在新增或需要替换时填写。</p>
              </div>

              <div>
                <label htmlFor="model" className="form-label">
                  服务商模型名称 <span className="field-code">model</span>
                  <span className="required">*</span>
                </label>
                <Input
                  id="model"
                  placeholder="gpt-4o / qwen3-plus"
                  value={llm?.model ?? ''}
                  onChange={(e) => {
                    const model = e.target.value;
                    setLlm((prev) => ({
                      ...prev,
                      model,
                      model_id: isAdd && !modelIdEdited ? model : prev.model_id,
                    }));
                  }}
                />
                <p className="field-hint">请求模型服务时传给服务商的模型名，例如 gpt-4o、qwen3-plus。</p>
              </div>
            </div>
          </div>

          {/* Identity section */}
          <div>
            <div className="dialog-section-title">PAI-RAG identity (optional)</div>
            <div className="space-y-3">
              <div>
                <label htmlFor="model_id" className="form-label">
                  PAI-RAG 模型标识 <span className="field-code">model_id</span>
                  <span className="required">*</span>
                </label>
                <Input
                  id="model_id"
                  placeholder={llm?.model || '默认与服务商模型名称一致'}
                  value={llm?.model_id ?? ''}
                  onChange={(e) => {
                    setModelIdEdited(true);
                    setLlm((prev) => ({ ...prev, model_id: e.target.value }));
                  }}
                />
                <p className="field-hint">默认跟随服务商模型名称；留空保存时会提交上方模型名称。若提示冲突，再改成 qwen3-plus-think / qwen3-plus-nothink 这类独立 ID。</p>
              </div>
            </div>
          </div>

          {/* Advanced section */}
          <div>
            <div className="dialog-section-title">Advanced</div>
            <div className="space-y-2">
              <div className="form-row-inline">
                <div className="form-row-inline-label">
                  <span className="title">{t('config.model.visionModel')}</span>
                  <span className="hint">支持图像输入</span>
                </div>
                <Switch
                  id="vision_support"
                  checked={llm?.vision_support ?? false}
                  onCheckedChange={(checked) =>
                    setLlm((prev) => ({ ...prev, vision_support: checked }))
                  }
                />
              </div>

              <div className="form-row-inline">
                <div className="form-row-inline-label">
                  <span className="title">{t('config.model.thinkingModel')}</span>
                  <span className="hint">开启链式思考 (Chain of Thought)</span>
                </div>
                <Switch
                  id="enable_thinking"
                  checked={llm?.enable_thinking ?? false}
                  onCheckedChange={(checked) =>
                    setLlm((prev) => ({ ...prev, enable_thinking: checked }))
                  }
                />
              </div>

              <div className="form-row-inline">
                <div className="form-row-inline-label">
                  <span className="title">{t('config.model.temperature')}</span>
                  <span className="hint">{t('config.model.temperatureHint')}</span>
                </div>
                <Input
                  id="temperature"
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  value={llm?.temperature ?? 0.1}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setLlm((prev) => ({
                      ...prev,
                      temperature: isNaN(val) || val < 0 ? 0.1 : val,
                    }));
                  }}
                  className="w-20 h-8 text-right"
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
            {displayIsAdd ? '创建 LLM 模型' : t('common.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
