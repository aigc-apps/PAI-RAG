import { useEffect, useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronRight } from "lucide-react";
import type { ReasoningStatus } from "../types";
import { cn } from "../lib/cn";
import { useI18n } from "../i18n";

export function CollapsibleReasoning({
  reasoning,
  status,
}: {
  reasoning: string;
  status: ReasoningStatus;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(status === "streaming");

  useEffect(() => {
    if (status === "streaming") setOpen(true);
    else if (status === "done") setOpen(false);
  }, [status]);

  if (!reasoning) return null;

  return (
    <Collapsible.Root
      open={open}
      onOpenChange={setOpen}
      className="mb-2 w-full"
    >
      <Collapsible.Trigger className="flex items-center gap-1 text-xs text-[var(--text-muted)] hover:text-[var(--text)] transition-colors">
        <ChevronRight
          className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-90")}
        />
        {status === "streaming" ? (
          <span className="shimmer-text font-medium">{t("reasoning.thinking")}</span>
        ) : (
          <span className="text-[var(--text-faint)]">{t("reasoning.thought")}</span>
        )}
      </Collapsible.Trigger>
      <Collapsible.Content className="mt-1 border-l-2 border-[var(--border-strong)] pl-3 text-xs text-[var(--text-muted)] whitespace-pre-wrap leading-6">
        {reasoning}
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
