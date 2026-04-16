'use client';

import { Button } from '@/components/ui/button';
import React, { useState } from 'react';
import { toast } from 'sonner';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export default function CacheConfig() {
  const { t } = useI18n();
  const [isClearing, setIsClearing] = useState(false);
  const { tenantFetch } = useTenantFetch();

  const handleClearMetadataSchemaCache = async () => {
    try {
      setIsClearing(true);
      const res = await tenantFetch(`/api/config/knowledgebases/metadata-schema-cache`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) throw new Error(t('config.cache.clearFailed'));
      const data = await res.json();
      const cleared = data?.data?.cleared ?? 0;
      toast.success(`${t('config.cache.clearSuccess')} (${cleared})`);
    } catch (err: any) {
      toast.error(err.message || t('config.cache.clearFailed'));
    } finally {
      setIsClearing(false);
    }
  };

  return (
    <div id="cache" className="settings-page">
      <div className="settings-page-header">
        <h1 className="page-title" suppressHydrationWarning>
          {t('config.cache.title')}
        </h1>
      </div>

      <div className="settings-page-content">
        <div className="space-y-6">
          <div>
            <div className="dialog-section-title mb-2">缓存项</div>
            <div className="form-row-inline">
              <div className="form-row-inline-label">
                <span className="title" suppressHydrationWarning>
                  {t('config.cache.metadataSchemaTitle')}
                </span>
                <span className="hint" suppressHydrationWarning>
                  {t('config.cache.metadataSchemaDescription')}
                </span>
              </div>
              <Button
                size="sm"
                variant="destructive"
                onClick={handleClearMetadataSchemaCache}
                disabled={isClearing}
              >
                <span suppressHydrationWarning>
                  {isClearing ? t('config.cache.clearing') : t('config.cache.clearCache')}
                </span>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
