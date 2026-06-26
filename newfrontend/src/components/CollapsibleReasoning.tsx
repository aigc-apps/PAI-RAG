import { useEffect, useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronRight, Brain } from "lucide-react";
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
      className="mb-2 rounded-md border border-gray-200 bg-gray-50 text-sm"
    >
      <Collapsible.Trigger className="flex w-full items-center gap-1 px-2 py-1 text-gray-500">
        <ChevronRight
          className={cn("h-4 w-4 transition-transform", open && "rotate-90")}
        />
        <Brain className="h-4 w-4" />
        <span>{status === "streaming" ? "Reasoning…" : "Reasoning"}</span>
      </Collapsible.Trigger>
      <Collapsible.Content className="whitespace-pre-wrap px-3 py-2 text-gray-600">
        {reasoning}
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
