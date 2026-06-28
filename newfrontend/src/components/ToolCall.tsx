import { useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { Loader2, Check, X, Wrench, ChevronRight } from "lucide-react";
import type { ToolUse } from "../types";
import { cn } from "../lib/cn";

function StatusIcon({ status }: { status: ToolUse["status"] }) {
  if (status === "running")
    return <Loader2 className="h-4 w-4 animate-spin text-[var(--accent)]" />;
  if (status === "done")
    return <Check className="h-4 w-4 text-emerald-500" />;
  return <X className="h-4 w-4 text-[var(--danger)]" />;
}

export function ToolCall({ tool }: { tool: ToolUse }) {
  const [open, setOpen] = useState(false);

  const statusWord =
    tool.status === "running" ? "running" : tool.status === "done" ? "done" : "error";

  return (
    <Collapsible.Root
      open={open}
      onOpenChange={setOpen}
      className="bg-[var(--tool-bg)] border border-[var(--border)] rounded-[var(--radius)] shadow-[var(--shadow-sm)] my-2 overflow-hidden"
    >
      <Collapsible.Trigger className="w-full flex items-center gap-2 px-3 py-2 text-sm">
        <StatusIcon status={tool.status} />
        <Wrench className="h-4 w-4 shrink-0 text-[var(--accent)]" />
        <span className="flex-1 font-medium text-left">{tool.name}</span>
        <span className="text-[var(--text-muted)]">{statusWord}</span>
        <ChevronRight
          className={cn(
            "h-4 w-4 shrink-0 text-[var(--text-muted)] transition-transform",
            open && "rotate-90"
          )}
        />
      </Collapsible.Trigger>
      <Collapsible.Content className="border-t border-[var(--border)] px-3 pb-3 pt-2">
        <div className="mb-2">
          <div className="mb-1 text-xs font-medium text-[var(--text-muted)]">Arguments</div>
          <pre className="font-mono text-xs bg-[var(--surface-2)] rounded-[var(--radius-sm)] p-2 whitespace-pre-wrap break-all">{tool.arguments}</pre>
        </div>
        {tool.status !== "running" && (
          <div>
            <div className="mb-1 text-xs font-medium text-[var(--text-muted)]">
              {tool.status === "error" ? "Error" : "Result"}
            </div>
            <pre className="font-mono text-xs bg-[var(--surface-2)] rounded-[var(--radius-sm)] p-2 max-h-64 overflow-auto whitespace-pre-wrap break-all">
              {tool.status === "error" ? tool.error : tool.output}
            </pre>
          </div>
        )}
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
