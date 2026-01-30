import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card'; // shadcn/ui 容器组件 [[1]]
import { ScrollArea } from '@/components/ui/scroll-area'; // 滚动区域支持 [[9]]
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export function JsonlViewer({ file_url }: { file_url: string }) {
  const { t } = useI18n();
  const { tenantFetch } = useTenantFetch();
  const [lines, setLines] = useState<Record<string, any>[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await tenantFetch(file_url);
        if (!response.ok) {
          setError('fileLoadFailed');
          return;
        }

        const text = await response.text();
        const parsedLines = text
          .split('\n')
          .filter(Boolean)
          .map((line) => JSON.parse(line));

        setLines(parsedLines);
      } catch (err) {
        setError('unknownError');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [file_url, tenantFetch]);

  if (loading) return <div>{t('common.loading')}</div>;
  if (error)
    return (
      <div className="text-red-500">
        {error === 'fileLoadFailed' ? t('knowledgebase.loadError') : t('messages.unknownError')}
      </div>
    );

  return (
    <ScrollArea className="pr-4">
      <div className="space-y-2">
        {lines.map((line, index) => (
          <pre key={index} className="bg-muted p-2 rounded-md overflow-x-auto">
            {JSON.stringify(line, null, 2)}
          </pre>
        ))}
      </div>
    </ScrollArea>
  );
}
