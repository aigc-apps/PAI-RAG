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
    <div className="mt-1 flex gap-2 text-[var(--text-faint)]">
      <button
        type="button"
        aria-label="Copy"
        onClick={copy}
        className="text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] rounded-md p-1.5 transition-colors"
      >
        {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
      </button>
      {onRegenerate && (
        <button
          type="button"
          aria-label="Regenerate"
          onClick={onRegenerate}
          className="text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] rounded-md p-1.5 transition-colors"
        >
          <RefreshCw className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}
