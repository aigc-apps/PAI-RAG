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
        className="hover:text-[var(--text)] transition-colors"
      >
        {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
      </button>
      {onRegenerate && (
        <button
          type="button"
          aria-label="Regenerate"
          onClick={onRegenerate}
          className="hover:text-[var(--text)] transition-colors"
        >
          <RefreshCw className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}
