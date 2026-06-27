import { useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { Loader2, Check, X, Wrench, ChevronRight } from "lucide-react";
import type { ToolUse } from "../types";
import { cn } from "../lib/cn";

function StatusIcon({ status }: { status: ToolUse["status"] }) {
  if (status === "running")
    return <Loader2 className="h-4 w-4 animate-spin text-[var(--text-muted)]" />;
  if (status === "done")
    return <Check className="h-4 w-4 text-green-600" />;
  return <X className="h-4 w-4 text-red-600" />;
}

export function ToolCall({ tool }: { tool: ToolUse }) {
  const [open, setOpen] = useState(false);

  const statusWord =
    tool.status === "running" ? "running" : tool.status === "done" ? "done" : "error";

  return (
    <Collapsible.Root
      open={open}
      onOpenChange={setOpen}
      className="my-2 w-full rounded-xl border border-[var(--border)] bg-[var(--tool-bg)] text-sm"
    >
      <Collapsible.Trigger className="flex w-full items-center gap-2 px-3 py-2 text-left">
        <ChevronRight
          className={cn(
            "h-4 w-4 shrink-0 text-[var(--text-muted)] transition-transform",
            open && "rotate-90"
          )}
        />
        <StatusIcon status={tool.status} />
        <Wrench className="h-4 w-4 shrink-0 text-[var(--text-muted)]" />
        <span className="flex-1 font-mono font-medium">{tool.name}</span>
        <span className="text-[var(--text-muted)]">{statusWord}</span>
      </Collapsible.Trigger>
      <Collapsible.Content className="border-t border-[var(--border)] px-3 pb-3 pt-2">
        <div className="mb-2">
          <div className="mb-1 text-xs font-medium text-[var(--text-muted)]">Arguments</div>
          <pre className="whitespace-pre-wrap break-all font-mono text-xs">{tool.arguments}</pre>
        </div>
        {tool.status !== "running" && (
          <div>
            <div className="mb-1 text-xs font-medium text-[var(--text-muted)]">
              {tool.status === "error" ? "Error" : "Result"}
            </div>
            <div className="max-h-64 overflow-auto font-mono text-xs">
              {tool.status === "error" ? tool.error : tool.output}
            </div>
          </div>
        )}
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
