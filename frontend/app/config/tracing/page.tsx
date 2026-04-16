'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { toast } from 'sonner';
import { ExternalLink } from 'lucide-react';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export default function TracingConfig() {
  const { t } = useI18n();
  const [endpoint, setEndpoint] = useState('');
  const [token, setToken] = useState('');
  const [serviceName, setServiceName] = useState('');
  const [traceEnabled, setTraceEnabled] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');
  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const res = await tenantFetch(`/api/config/trace`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });
        if (!res.ok) throw new Error(t('config.loadError'));
        const data = (await res.json()).data;
        setEndpoint(data['endpoint'] || '');
        setToken(data['token'] || '');
        setServiceName(data['service_name'] || '');
        setTraceEnabled(data['enabled'] || false);
      } catch (err: any) {
        toast.error(err.message);
      }
    };
    fetchConfig();
  }, []);

  const handleSave = async () => {
    if (!endpoint || !token || !serviceName) {
      setError(t('config.tracing.fieldsRequired'));
      return;
    }
    try {
      setIsSaving(true);
      setError('');
      const res = await tenantFetch(`/api/config/trace`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          endpoint: endpoint,
          token: token,
          service_name: serviceName,
          enabled: traceEnabled,
        }),
      });
      if (!res.ok) throw new Error(t('config.tracing.saveFailed'));
      toast.success(t('config.tracing.saveSuccess'));
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div id="tracing" className="settings-page">
      <div className="settings-page-header">
        <h1 className="page-title">{t('config.tracing.aliyunTracingConfig')}</h1>
        <Button size="sm" onClick={handleSave} disabled={isSaving}>
          {isSaving ? t('config.tracing.saving') : t('config.tracing.saveTracingConfig')}
        </Button>
      </div>

      <div className="settings-page-content">
        <div className="space-y-6">
          <div>
            <a
              href="https://help.aliyun.com/zh/opentelemetry/quick-start?spm=a2c4g.11186623.help-menu-90275.d_1.15c45dc7tG5ukV#prereq-3jq-3as-xo9"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-primary hover:underline inline-flex items-center gap-1"
            >
              {t('config.tracing.howToGetInfo')}
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          <div>
            <div className="dialog-section-title mb-2">OpenTelemetry 端点</div>
            <div className="space-y-4">
              <div>
                <label htmlFor="endpoint" className="form-label">
                  {t('config.tracing.endpoint')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="endpoint"
                  value={endpoint}
                  onChange={(e) => setEndpoint(e.target.value)}
                  placeholder={t('config.tracing.endpointPlaceholder')}
                />
              </div>

              <div>
                <label htmlFor="token" className="form-label">
                  Token<span className="required">*</span>
                </label>
                <Input
                  id="token"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder={t('config.tracing.tokenPlaceholder')}
                />
              </div>

              <div>
                <label htmlFor="serivceName" className="form-label">
                  {t('config.tracing.serviceName')}
                  <span className="required">*</span>
                </label>
                <Input
                  id="serivceName"
                  value={serviceName}
                  onChange={(e) => setServiceName(e.target.value)}
                  placeholder={t('config.tracing.serviceNamePlaceholder')}
                />
              </div>
            </div>
          </div>

          <div>
            <div className="dialog-section-title mb-2">追踪开关</div>
            <div className="form-row-inline">
              <div className="form-row-inline-label">
                <span className="title">{t('config.tracing.isEnabled')}</span>
                <span className="hint">开启后系统请求将上报至配置的 OpenTelemetry Endpoint</span>
              </div>
              <Switch
                id="traceEnabled"
                checked={traceEnabled || false}
                onCheckedChange={(v) => setTraceEnabled(v === true)}
              />
            </div>
          </div>

          {error && <p className="text-destructive text-sm">{error}</p>}
        </div>
      </div>
    </div>
  );
}
