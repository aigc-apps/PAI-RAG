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
    <div id="cache">
      <div className="transition-colors rounded-lg p-4 overflow-hidden duration-200">
        <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-gray-200 rounded-xl bg-gray-50">
          <h2 className="text-xl font-medium text-gray-800" suppressHydrationWarning>
            {t('config.cache.title')}
          </h2>

          <div className="max-w-xl w-full mx-auto mt-8 space-y-6">
            <div className="border rounded-lg p-6 bg-white">
              <h3 className="text-base font-medium text-gray-700 mb-2" suppressHydrationWarning>
                {t('config.cache.metadataSchemaTitle')}
              </h3>
              <p className="text-sm text-gray-500 mb-4" suppressHydrationWarning>
                {t('config.cache.metadataSchemaDescription')}
              </p>
              <Button
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
