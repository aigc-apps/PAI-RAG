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
    <div className="mt-2 flex gap-1 text-[var(--text-faint)]">
      <button
        type="button"
        aria-label="Copy"
        onClick={copy}
        className="rounded-[var(--radius-sm)] p-1 hover:bg-[var(--surface-2)] hover:text-[var(--text)] transition-colors"
      >
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
      {onRegenerate && (
        <button
          type="button"
          aria-label="Regenerate"
          onClick={onRegenerate}
          className="rounded-[var(--radius-sm)] p-1 hover:bg-[var(--surface-2)] hover:text-[var(--text)] transition-colors"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}
