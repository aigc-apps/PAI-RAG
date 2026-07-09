import { Copy, RefreshCw, Check } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { copyText } from "../lib/clipboard";
import type { ChatMessage } from "../types";

export function MessageControls({
  text,
  usage,
  onRegenerate,
}: {
  text: string;
  usage?: ChatMessage["usage"];
  onRegenerate?: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    if (await copyText(text)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } else {
      toast.error("复制失败");
    }
  };

  return (
    <div className="mt-2 flex gap-1">
      <button
        type="button"
        aria-label="复制"
        title="复制"
        onClick={copy}
        className="icon-btn p-1"
      >
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
      {onRegenerate && (
        <button
          type="button"
          aria-label="重新生成"
          title="重新生成"
          onClick={onRegenerate}
          className="icon-btn p-1"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      )}
      {usage && usage.total > 0 && (
        <span
          className="ml-1 self-center text-xs text-[var(--text-faint)]"
          title={`输入 ${usage.input.toLocaleString()} · 输出 ${usage.output.toLocaleString()} tokens`}
        >
          {usage.total.toLocaleString()} tokens
        </span>
      )}
    </div>
  );
}
