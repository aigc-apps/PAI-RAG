import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card'; // shadcn/ui 容器组件 [[1]]
import { ScrollArea } from '@/components/ui/scroll-area'; // 滚动区域支持 [[9]]
import { useTenantFetch } from '@/hooks/use-tenant-fetch';

export function JsonlViewer({ file_url }: { file_url: string }) {
  const [lines, setLines] = useState<Record<string, any>[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { tenantFetch } = useTenantFetch();
        const response = await tenantFetch(file_url);
        if (!response.ok) throw new Error('文件加载失败');

        const text = await response.text();
        const parsedLines = text
          .split('\n')
          .filter(Boolean)
          .map((line) => JSON.parse(line));

        setLines(parsedLines);
      } catch (err) {
        setError(err instanceof Error ? err.message : '未知错误');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [file_url]);

  if (loading) return <div>加载中...</div>;
  if (error) return <div className="text-red-500">{error}</div>;

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
