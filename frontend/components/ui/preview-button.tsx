"use client";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogTrigger,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";

interface PreviewButtonProps {
  markdownContent: string;
}

export function PreviewButton({ markdownContent }: PreviewButtonProps) {
  const [open, setOpen] = useState(false);
  const [content, setContent] = useState("");

  // 模拟异步加载Markdown内容
  useEffect(() => {
    if (open && markdownContent) {
      setContent(markdownContent);
    }
  }, [open, markdownContent]);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="link" className="text-sm text-blue-600 p-0">
          预览
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Markdown 预览</DialogTitle>
        </DialogHeader>
        <ScrollArea className="grid gap-2 py-2 max-h-[600px] max-w-[800px]">
          <div className="markdown-content">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
