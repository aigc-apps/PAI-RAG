import { Copy, RefreshCw, Check } from "lucide-react";
import { useState } from "react";

export function MessageControls({
  text,
  onRegenerate,
}: {
  text: string;
  onRegenerate?: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
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
    </div>
  );
}
