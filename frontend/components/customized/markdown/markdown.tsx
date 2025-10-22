// components/markdown-renderer.tsx
"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Check, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";

interface MarkdownRendererProps {
  content: string;
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
            table: ({ node, ...props }) => (
              <div className="overflow-x-auto">
                <table className="min-w-full border-collapse border border-gray-300" {...props} />
              </div>
            ),
            th: ({ node, ...props }) => (
              <th
                className="border border-gray-300 px-4 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider bg-gray-50"
                {...props}
              />
            ),
            tbody: ({ node, ...props }) => (
              <tbody className="bg-white divide-y divide-gray-550" {...props} />
            ),
            td: ({ node, ...props }) => (
              <td className="border border-gray-300 px-4 py-3 whitespace-nowrap text-sm text-gray-900" {...props} />
            ),
          // 处理行内代码
          p: ({ node, children, ...props }) => (
            <p className="leading-relaxed my-2" {...props}>
              {children}
            </p>
          ),
          // 处理代码块
          code({ node, className, children, ...props }) {
            // 代码块：提取语言
            const match = /language-(\w+)/.exec(className || "");
            const language = match?.[1] || "text";
            const codeString = String(children).replace(/\n$/, "");

            // 语言别名映射（可选）
            const langMap: Record<string, string> = {
              js: "javascript",
              ts: "typescript",
              py: "python",
              sh: "bash",
              yml: "yaml",
              md: "markdown",
            };
            const resolvedLang = langMap[language] || language;

            // 复制功能
            const [copied, setCopied] = React.useState(false);
            const handleCopy = () => {
              navigator.clipboard.writeText(codeString);
              setCopied(true);
              setTimeout(() => setCopied(false), 2000);
            };

            return (
              <div className="relative my-4 group">
                {/* 顶部工具栏 */}
                <div className="flex items-center justify-between px-4 py-2 bg-gray-800 text-gray-200 text-xs rounded-t-md">
                  <span className="font-mono">{resolvedLang}</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleCopy}
                    className="h-6 px-2 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    {copied ? (
                      <>
                        <Check className="h-3 w-3 mr-1" />
                        已复制
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3 mr-1" />
                        复制
                      </>
                    )}
                  </Button>
                </div>

                {/* 代码高亮区域 */}
                <SyntaxHighlighter
                  language={resolvedLang}
                  style={oneDark}
                  customStyle={{
                    margin: 0,
                    borderRadius: "0 0 6px 6px",
                    fontSize: "14px",
                    lineHeight: "1.5",
                  }}
                  wrapLines={true}
                  showLineNumbers={true}
                  lineNumberStyle={{
                    opacity: 0.5,
                    paddingRight: "16px",
                    minWidth: "32px",
                    textAlign: "right" as const,
                  }}
                >
                  {codeString}
                </SyntaxHighlighter>
              </div>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}