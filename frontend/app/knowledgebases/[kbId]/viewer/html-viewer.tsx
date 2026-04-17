import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useI18n } from '@/app/providers/i18n';
import { PageLoading } from '@/components/ui/loading';
export function HtmlViewer({ file_url }: { file_url: string }) {
  const { t } = useI18n();
  const [htmlContent, setHtmlContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHtml = async () => {
      try {
        const response = await fetch(file_url);
        if (!response.ok) {
          setError('fileLoadFailed');
          return;
        }
        const text = await response.text();
        setHtmlContent(text);
      } catch (err) {
        setError('unknownError');
      } finally {
        setLoading(false);
      }
    };

    fetchHtml();
  }, [file_url]);

  if (loading) return <PageLoading />;
  if (error)
    return (
      <div className="text-red-500">
        {error === 'fileLoadFailed' ? t('knowledgebase.loadError') : t('messages.unknownError')}
      </div>
    );

  return (
    <ScrollArea className="pr-4">
      <div
        className="bg-muted p-4 rounded-md"
        dangerouslySetInnerHTML={{ __html: htmlContent }}
      />
    </ScrollArea>
  );
}
