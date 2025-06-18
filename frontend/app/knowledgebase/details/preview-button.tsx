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
import { ScrollArea } from "@/components/ui/scroll-area";
import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { Loader2 } from "lucide-react";

export function PreviewButton({
  kbId,
  fileId,
}: {
  kbId: string;
  fileId: string;
}) {
  const [open, setOpen] = useState(false);
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadContent = async () => {
    setLoading(true);
    try {
      // // 调用preview接口获取内容
      // const response = await fetch(`/api/knowledgebases/{kb_name}/files/${fileId}/preview`);
      // if (!response.ok) throw new Error("加载失败");
      // const text = await response.text();
      console.log("加载文件内容", kbId, fileId);
      // 模拟加载内容
      const text =
        '# 系统文档指南\n\n## 简介\n\n这是使用现代样式渲染的 Markdown 文档示例。以下展示了各种格式的渲染效果：\n\n### 标题层级\n\n#### 三级标题下的四级标题\n\n- 支持无序列表\n\n- 支持有序列表\n\n1. 嵌套有序列表\n\n2. 第二项\n\n**强调文本** 和 `行内代码` 示例\n\n```python\n\n# 代码块示例\n\ndef hello():\n\nprint("现代 Markdown 样式")';
      setContent(text);
    } catch (err) {
      setError("无法加载文件内容");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      {/* <DialogTrigger asChild>
        <Button
          variant="link"
          className="text-sm text-blue-600 p-0"
          onClick={loadContent}
        >
          {loading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            "预览"
          )}
        </Button>
      </DialogTrigger> */}
      <DialogTrigger asChild>
        <Button
          variant="link"
          className="text-sm text-blue-600 p-0"
          onClick={loadContent}
        >
          预览
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Markdown 预览</DialogTitle>
          <DialogDescription>
            预览文件markdown格式的解析内容。
          </DialogDescription>
        </DialogHeader>
        <ScrollArea className="grid gap-2 py-2 max-h-[600px] max-w-[800px]">
          {error ? (
            <div className="text-red-500">{error}</div>
          ) : loading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
          )}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
