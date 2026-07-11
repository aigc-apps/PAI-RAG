import { Copy, RefreshCw, Check } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { copyText } from "../lib/clipboard";
import type { ChatMessage } from "../types";
import { useI18n } from "../i18n";

export function MessageControls({
  text,
  usage,
  onRegenerate,
}: {
  text: string;
  usage?: ChatMessage["usage"];
  onRegenerate?: () => void;
}) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    if (await copyText(text)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } else {
      toast.error(t("common.copyFailed"));
    }
  };

  return (
    <div className="mt-2 flex gap-1">
      <button
        type="button"
        aria-label={t("common.copy")}
        title={t("common.copy")}
        onClick={copy}
        className="icon-btn p-1"
      >
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
      {onRegenerate && (
        <button
          type="button"
          aria-label={t("msg.regenerate")}
          title={t("msg.regenerate")}
          onClick={onRegenerate}
          className="icon-btn p-1"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      )}
      {usage && usage.total > 0 && (
        <span
          className="ml-1 self-center text-xs text-[var(--text-faint)]"
          title={t("msg.usageTitle", {
            input: usage.input.toLocaleString(),
            output: usage.output.toLocaleString(),
          })}
        >
          {t("msg.tokens", { count: usage.total.toLocaleString() })}
        </span>
      )}
    </div>
  );
}
