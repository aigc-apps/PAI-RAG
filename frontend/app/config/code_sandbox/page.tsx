'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { toast } from 'sonner';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

const MASK_API_KEY = '******'

export default function CodeSandboxConfig() {
  const { t } = useI18n();
  const [isEnabled, setIsEnabled] = useState(false);
  const [configType, setConfigType] = useState('aliyun-fc'); // 目前仅支持 aliyun-fc
  const [aliyunId, setAliyunId] = useState('');
  const [interpreterId, setInterpreterId] = useState('');
  const [interpreterName, setInterpreterName] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [hasApiKey, setHasApiKey] = useState(false); // 标记是否存在 API key
  const [timeoutDefault, setTimeoutDefault] = useState(50);
  const [isSaving, setIsSaving] = useState(false);
  const { tenantFetch } = useTenantFetch();
  // 初始化加载配置
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const res = await tenantFetch(`/api/config/code_sandbox`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) throw new Error(t('config.codeSandbox.loadError'));

        const response = await res.json();
        const config = response?.data;

        if (config) {
          setIsEnabled(config.enabled || false);
          setConfigType(config.type || 'aliyun-fc');
          setAliyunId(config.aliyun_id || '');
          setInterpreterId(config.interpreter_id || '');
          setInterpreterName(config.interpreter_name || '');
          const hasKey = !!config.api_key;
          setHasApiKey(hasKey);
          setApiKey(hasKey ? MASK_API_KEY : '');
          setTimeoutDefault(config.timeout_default || 50);
        }
      } catch (err: any) {
        toast.error(err.message);
      }
    };

    fetchConfig();
  }, []);

  const handleSave = async () => {
    try {
      setIsSaving(true);

      const payload: Record<string, any> = {
        enabled: isEnabled,
        type: configType,
        timeout_default: timeoutDefault,
      };

      // 仅当用户输入了值才提交（避免覆盖已有值为空）
      if (aliyunId) payload.aliyun_id = aliyunId;
      if (interpreterId) payload.interpreter_id = interpreterId;
      if (interpreterName) payload.interpreter_name = interpreterName;
      // 如果用户输入的是占位符，则不更新 API key（保持原值）；否则更新
      if (apiKey !== MASK_API_KEY) {
        payload.api_key = apiKey || null;
      }
      if (isEnabled) payload.enabled = isEnabled;
      if (timeoutDefault) payload.timeout_default = timeoutDefault;


      const res = await tenantFetch(`/api/config/code_sandbox`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(t('config.codeSandbox.saveFailed'));

      toast.success(t('config.codeSandbox.saveSuccess'));
      
      // 重新加载配置以更新状态
      const refreshRes = await tenantFetch(`/api/config/code_sandbox`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      if (refreshRes.ok) {
        const refreshResponse = await refreshRes.json();
        const refreshConfig = refreshResponse?.data;
        if (refreshConfig) {
          const hasKey = !!refreshConfig.api_key;
          setHasApiKey(hasKey);
          setApiKey(hasKey ? MASK_API_KEY : '');
        }
      }
    } catch (err: any) {
      toast.warning(err.message);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div id="codesandbox">
      <div className="transition-colors rounded-lg p-4 overflow-hidden duration-200">
        <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          <h2 className="text-xl font-medium text-gray-800">{t('config.codeSandbox.title')}</h2>

          <div className="grid gap-4 py-4 w-full max-w-2xl">
            {/* 启用开关 */}
            <div className="grid grid-cols-4 items-center gap-4">
              <Label className="text-right">{t('config.codeSandbox.enableSandbox')}</Label>
              <div className="col-span-3 flex items-center space-x-2">
                <Switch
                  checked={isEnabled}
                  onCheckedChange={setIsEnabled}
                />
                <span className="text-sm text-gray-600">
                  {isEnabled ? t('config.codeSandbox.enabled') : t('config.codeSandbox.disabled')}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-4 items-center gap-4">
              <Label className="text-right">{t('config.codeSandbox.sandboxType')}</Label>
              <div className="col-span-3">
                <Select value={configType} onValueChange={setConfigType} disabled>
                  <SelectTrigger>
                    <SelectValue placeholder={t('config.codeSandbox.selectSandboxType')} />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="aliyun-fc">{t('config.codeSandbox.aliyunFcSandbox')}</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-gray-500 mt-1">{t('config.codeSandbox.currentlyOnlySupported')}</p>
              </div>
            </div>

            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="aliyun_id" className="text-right">
                {t('config.codeSandbox.aliyunId')}
              </Label>
              <div className="col-span-3">
                <Input
                  id="aliyun_id"
                  value={aliyunId}
                  onChange={(e) => setAliyunId(e.target.value)}
                  placeholder={t('config.codeSandbox.aliyunIdPlaceholder')}
                />
              </div>
            </div>

            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="interpreter_id" className="text-right">
                {t('config.codeSandbox.interpreterId')}
              </Label>
              <div className="col-span-3">
                <Input
                  id="interpreter_id"
                  value={interpreterId}
                  onChange={(e) => setInterpreterId(e.target.value)}
                  placeholder={t('config.codeSandbox.interpreterIdPlaceholder')}
                />
              </div>
            </div>

            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="interpreter_name" className="text-right">
                {t('config.codeSandbox.interpreterName')}
              </Label>
              <div className="col-span-3">
                <Input
                  id="interpreter_name"
                  value={interpreterName}
                  onChange={(e) => setInterpreterName(e.target.value)}
                  placeholder={t('config.codeSandbox.interpreterNamePlaceholder')}
                />
              </div>
            </div>

            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="api_key" className="text-right">
                {t('config.codeSandbox.apiKey')}
              </Label>
              <div className="col-span-3">
                <Input
                  id="api_key"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={t('config.codeSandbox.apiKeyPlaceholder')}
                />
              </div>
            </div>

            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="timeout_default" className="text-right">
                {t('config.codeSandbox.defaultTimeout')}({timeoutDefault})
              </Label>
              <div className="col-span-3">
                <Slider
                  value={[timeoutDefault]}
                  max={300}
                  min={10}
                  step={5}
                  onValueChange={(value) => setTimeoutDefault(value[0])}
                />
              </div>
            </div>
          </div>

          <Button
            onClick={handleSave}
            disabled={isSaving}
            className="mt-4 px-4 py-2 text-white rounded-lg transition-colors"
          >
            {isSaving ? t('common.saving') : t('config.codeSandbox.saveConfig')}
          </Button>
        </div>
      </div>
    </div>
  );
}