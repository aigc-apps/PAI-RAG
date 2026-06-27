import { useEffect, useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronRight } from "lucide-react";
import type { ReasoningStatus } from "../types";
import { cn } from "../lib/cn";

export function CollapsibleReasoning({
  reasoning,
  status,
}: {
  reasoning: string;
  status: ReasoningStatus;
}) {
  const [open, setOpen] = useState(status === "streaming");

  // Auto-open while streaming, auto-collapse when reasoning finishes.
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
      <Collapsible.Trigger className="flex items-center gap-1 text-sm text-[var(--text-muted)] hover:text-[var(--text)]">
        <ChevronRight
          className={cn("h-4 w-4 transition-transform", open && "rotate-90")}
        />
        <span>{status === "streaming" ? "Thinking…" : "Thought"}</span>
      </Collapsible.Trigger>
      <Collapsible.Content className="mt-1 border-l-2 border-[var(--border)] pl-3 text-sm text-[var(--text-muted)] whitespace-pre-wrap">
        {reasoning}
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
