import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { useI18n } from '@/app/providers/i18n';
import { PageLoading } from '@/components/ui/loading';
export function MarkdownViewer({ file_url }: { file_url: string }) {
  const { t } = useI18n();
  const [markdown, setMarkdown] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMarkdown = async () => {
      try {
        const response = await fetch(file_url);
        if (!response.ok) {
          setError('fileLoadFailed');
          return;
        }
        const text = await response.text();
        setMarkdown(text);
      } catch (err) {
        setError('unknownError');
      } finally {
        setLoading(false);
      }
    };

    fetchMarkdown();
  }, [file_url]);

  if (loading) return <PageLoading />;
  if (error)
    return (
      <div className="text-red-500">
        {error === 'fileLoadFailed' ? t('knowledgebase.loadError') : t('messages.unknownError')}
      </div>
    );

  return (
    <div className="markdown-content">
      <ReactMarkdown>{markdown}</ReactMarkdown>
    </div>
  );
}
