'use client';

import { Button } from '@/components/ui/button';
import React, { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { toast } from 'sonner';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export default function TracingConfig() {
  const { t } = useI18n();
  const [endpoint, setEndpoint] = useState('');
  const [token, setToken] = useState('');
  const [serviceName, setServiceName] = useState('');
  const [traceEnabled, setTraceEnabled] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const [error, setError] = useState('');

  const { tenantFetch } = useTenantFetch();
  // Initialize and load configuration
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setIsLoading(true);

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
      } finally {
        setIsLoading(false);
      }
    };

    fetchConfig();
  }, []);
  // Save configuration
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
    <div id="tracing">
      <div
        className={
          'transition-colors rounded-lg p-4 overflow-hidden duration-200'
        }
      >
        <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          <h2 className="text-xl font-medium text-gray-800">
            {t('config.tracing.aliyunTracingConfig')}
          </h2>
          <a
            href="https://help.aliyun.com/zh/opentelemetry/quick-start?spm=a2c4g.11186623.help-menu-90275.d_1.15c45dc7tG5ukV#prereq-3jq-3as-xo9"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline text-sm"
          >
            {t('config.tracing.howToGetInfo')}
          </a>

          <div className="grid gap-4 py-4 max-w-xl w-full mx-auto">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="endpoint">{t('config.tracing.endpoint')}</Label>
              <div className="col-span-3 flex items-center">
                <Input
                  id="endpoint"
                  value={endpoint}
                  onChange={(e) => setEndpoint(e.target.value)}
                  placeholder={t('config.tracing.endpointPlaceholder')}
                  className="col-span-3"
                />
              </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="token">Token</Label>
              <div className="col-span-3 flex items-center">
                <Input
                  id="token"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder={t('config.tracing.tokenPlaceholder')}
                  className="col-span-3"
                />
              </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="serivceName">{t('config.tracing.serviceName')}</Label>
              <div className="col-span-3 flex items-center">
                <Input
                  id="serivceName"
                  value={serviceName}
                  onChange={(e) => setServiceName(e.target.value)}
                  placeholder={t('config.tracing.serviceNamePlaceholder')}
                  className="col-span-3"
                />
              </div>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="traceEnabled">{t('config.tracing.isEnabled')}</Label>
              <Checkbox
                id="traceEnabled"
                checked={traceEnabled || false}
                onCheckedChange={(checkedState) => {
                  const isChecked = checkedState === true;
                  setTraceEnabled(isChecked);
                }}
                className="col-span-3"
              />
            </div>
          </div>
          <Button
            onClick={handleSave}
            disabled={isSaving}
            className="mt-4 px-4 py-2 text-white rounded-lg transition-colors"
          >
            {isSaving ? t('config.tracing.saving') : t('config.tracing.saveTracingConfig')}
          </Button>
          {error && <p className="text-red-500 mt-2">{error}</p>}
        </div>
      </div>
    </div>
  );
}
