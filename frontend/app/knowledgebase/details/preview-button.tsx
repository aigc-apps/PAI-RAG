"use client";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogTrigger,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { useState, useEffect } from "react";
import { MarkdownViewer } from "@/app/knowledgebase/details/viewer/markdown-viewer";
import { JsonlViewer } from "@/app/knowledgebase/details/viewer/jsonl-viewer";
import { HtmlViewer } from "@/app/knowledgebase/details/viewer/html-viewer";

interface KnowledgeBaseFile {
  id: string;
  file_name: string;
  file_size: string;
  file_extension: string;
  file_metadata: {
    file_url: string;
  };
  updated_at: string;
}

export function PreviewButton({
  kbId,
  fileId,
}: {
  kbId: string;
  fileId: string;
}) {
  const [open, setOpen] = useState(false);
  const [kbfile, setKbFile] = useState<KnowledgeBaseFile>(); // 文件详情
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadContent = async () => {
    setLoading(true);
    try {
      const API_BASE =
        process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8688";
      const res = await fetch(
        `${API_BASE}/v1/config/knowledgebases/${kbId}/files/${fileId}`,
      );
      if (!res.ok) throw new Error("获取知识库文件失败");
      const json_data = await res.json();
      const kb_file_data = json_data.data;

      setKbFile(kb_file_data); // 更新状态
      console.log("知识库文件详情数据:", kb_file_data);
    } catch (err: any) {
      setError(err || "加载失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="link"
          className="text-sm text-blue-600 p-0"
          onClick={loadContent}
        >
          文件预览
        </Button>
      </DialogTrigger>
      <DialogContent className="flex flex-col h-[calc(100%-10rem)] !max-w-[calc(100%-20rem)]">
        <DialogHeader className="flex-none h-1/10">
          <DialogTitle>{kbfile?.file_name}</DialogTitle>
          <DialogDescription>文件预览</DialogDescription>
        </DialogHeader>
        <div className="flex-grow overflow-y-auto">
          {kbfile?.file_extension === ".pdf" ? (
            <iframe
              src={kbfile?.file_metadata.file_url}
              width="100%"
              height="100%"
              title="PDF预览"
            ></iframe>
          ) : kbfile?.file_extension === ".jpg" ||
            kbfile?.file_extension === ".png" ||
            kbfile?.file_extension === ".jpeg" ? (
            <img
              src={kbfile?.file_metadata.file_url}
              width="100%"
              height="100%"
              title="图片预览"
            ></img>
          ) : kbfile?.file_extension === ".docx" ||
            kbfile?.file_extension === ".xlsx" ||
            kbfile?.file_extension === ".pptx" ? (
            <iframe
              src={`https://view.officeapps.live.com/op/embed.aspx?src=${encodeURIComponent(
                String(kbfile?.file_metadata.file_url),
              )}`}
              width="100%"
              height="100%"
              title="文件预览"
            />
          ) : kbfile?.file_extension === ".md" ||
            kbfile?.file_extension === ".txt" ? (
            <MarkdownViewer file_url={kbfile?.file_metadata.file_url} />
          ) : kbfile?.file_extension === ".jsonl" ? (
            <JsonlViewer file_url={kbfile?.file_metadata.file_url} />
          ) : kbfile?.file_extension === ".html" ? (
            <HtmlViewer file_url={kbfile?.file_metadata.file_url} />
          ) : (
            <div>
              暂不支持此格式文件的在线预览，请直接下载查看
              <a
                href={kbfile?.file_metadata.file_url}
                className="text-blue-500 hover:underline"
              >
                下载文件
              </a>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
