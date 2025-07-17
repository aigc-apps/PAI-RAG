import React, { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
export function HtmlViewer({ file_url }: { file_url: string }) {
  const [htmlContent, setHtmlContent] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(file_url)
      .then((response) => {
        if (!response.ok) throw new Error("加载失败");
        return response.text();
      })
      .then((text) => {
        setHtmlContent(text);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [file_url]);

  if (loading) return <div>加载中...</div>;
  if (error) return <div className="text-red-500">{error}</div>;

  return (
    <ScrollArea className="pr-4">
      <div
        className="bg-muted p-4 rounded-md"
        dangerouslySetInnerHTML={{ __html: htmlContent }}
      />
    </ScrollArea>
  );
}
