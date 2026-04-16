'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { toast } from 'sonner';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

const MASK_API_KEY = '******';

export default function CodeSandboxConfig() {
  const { t } = useI18n();
  const [isEnabled, setIsEnabled] = useState(false);
  const [configType, setConfigType] = useState('aliyun-fc');
  const [aliyunId, setAliyunId] = useState('');
  const [interpreterId, setInterpreterId] = useState('');
  const [interpreterName, setInterpreterName] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [hasApiKey, setHasApiKey] = useState(false);
  const [timeoutDefault, setTimeoutDefault] = useState(50);
  const [isSaving, setIsSaving] = useState(false);
  const { tenantFetch } = useTenantFetch();

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
      if (aliyunId) payload.aliyun_id = aliyunId;
      if (interpreterId) payload.interpreter_id = interpreterId;
      if (interpreterName) payload.interpreter_name = interpreterName;
      if (apiKey !== MASK_API_KEY) {
        payload.api_key = apiKey || null;
      }

      const res = await tenantFetch(`/api/config/code_sandbox`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(t('config.codeSandbox.saveFailed'));
      toast.success(t('config.codeSandbox.saveSuccess'));

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
    <div id="codesandbox" className="settings-page">
      <div className="settings-page-header">
        <h1 className="page-title">{t('config.codeSandbox.title')}</h1>
        <Button size="sm" onClick={handleSave} disabled={isSaving}>
          {isSaving ? t('common.saving') : t('config.codeSandbox.saveConfig')}
        </Button>
      </div>

      <div className="settings-page-content">
        <div className="space-y-6">
          <div>
            <div className="dialog-section-title mb-2">沙箱状态</div>
            <div className="form-row-inline">
              <div className="form-row-inline-label">
                <span className="title">{t('config.codeSandbox.enableSandbox')}</span>
                <span className="hint">
                  {isEnabled
                    ? t('config.codeSandbox.enabled')
                    : t('config.codeSandbox.disabled')}
                </span>
              </div>
              <Switch checked={isEnabled} onCheckedChange={setIsEnabled} />
            </div>
          </div>

          <div>
            <div className="dialog-section-title mb-2">沙箱类型</div>
            <Select value={configType} onValueChange={setConfigType} disabled>
              <SelectTrigger>
                <SelectValue placeholder={t('config.codeSandbox.selectSandboxType')} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="aliyun-fc">
                  {t('config.codeSandbox.aliyunFcSandbox')}
                </SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-1">
              {t('config.codeSandbox.currentlyOnlySupported')}
            </p>
          </div>

          <div>
            <div className="dialog-section-title mb-2">Aliyun FC 凭证</div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="aliyun_id" className="form-label">
                  {t('config.codeSandbox.aliyunId')}
                </label>
                <Input
                  id="aliyun_id"
                  value={aliyunId}
                  onChange={(e) => setAliyunId(e.target.value)}
                  placeholder={t('config.codeSandbox.aliyunIdPlaceholder')}
                />
              </div>
              <div>
                <label htmlFor="api_key" className="form-label">
                  {t('config.codeSandbox.apiKey')}
                </label>
                <Input
                  id="api_key"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={t('config.codeSandbox.apiKeyPlaceholder')}
                />
              </div>
              <div>
                <label htmlFor="interpreter_id" className="form-label">
                  {t('config.codeSandbox.interpreterId')}
                </label>
                <Input
                  id="interpreter_id"
                  value={interpreterId}
                  onChange={(e) => setInterpreterId(e.target.value)}
                  placeholder={t('config.codeSandbox.interpreterIdPlaceholder')}
                />
              </div>
              <div>
                <label htmlFor="interpreter_name" className="form-label">
                  {t('config.codeSandbox.interpreterName')}
                </label>
                <Input
                  id="interpreter_name"
                  value={interpreterName}
                  onChange={(e) => setInterpreterName(e.target.value)}
                  placeholder={t('config.codeSandbox.interpreterNamePlaceholder')}
                />
              </div>
            </div>
          </div>

          <div>
            <div className="dialog-section-title mb-2">执行参数</div>
            <div className="form-row-inline">
              <div className="form-row-inline-label">
                <span className="title">
                  {t('config.codeSandbox.defaultTimeout')}({timeoutDefault}s)
                </span>
                <span className="hint">单次代码执行超时时间 (10–300 秒)</span>
              </div>
              <div className="w-48">
                <Slider
                  value={[timeoutDefault]}
                  max={300}
                  min={10}
                  step={5}
                  onValueChange={(v) => setTimeoutDefault(v[0])}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
