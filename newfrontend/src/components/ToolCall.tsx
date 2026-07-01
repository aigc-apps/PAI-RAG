import { useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { Check, X, ChevronRight } from "lucide-react";
import type { ToolUse } from "../types";
import { cn } from "../lib/cn";

function StatusDot({ status }: { status: ToolUse["status"] }) {
  if (status === "running")
    return <span className="h-2 w-2 rounded-full bg-[var(--accent)] pulse-dot" />;
  if (status === "done")
    return <span className="h-2 w-2 rounded-full bg-[var(--success)]" />;
  return <span className="h-2 w-2 rounded-full bg-[var(--danger)]" />;
}

function StatusIcon({ status }: { status: ToolUse["status"] }) {
  if (status === "done")
    return <Check className="h-3.5 w-3.5 text-[var(--success)]" />;
  if (status === "error")
    return <X className="h-3.5 w-3.5 text-[var(--danger)]" />;
  return null;
}

export function ToolCall({ tool }: { tool: ToolUse }) {
  const [open, setOpen] = useState(false);

  const statusWord =
    tool.status === "running" ? "running" : tool.status === "done" ? "done" : "error";

  return (
    <Collapsible.Root
      open={open}
      onOpenChange={setOpen}
      className="my-1 overflow-hidden rounded-[var(--radius-sm)] border border-[var(--border)] bg-transparent"
    >
      <Collapsible.Trigger className="w-full flex items-center gap-2 px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] transition-colors">
        <StatusDot status={tool.status} />
        <span className="font-mono text-xs font-medium text-[var(--text)]">{tool.name}</span>
        <span className="text-xs text-[var(--text-faint)] ml-1">{statusWord}</span>
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 ml-auto shrink-0 text-[var(--text-faint)] transition-transform",
            open && "rotate-90"
          )}
        />
      </Collapsible.Trigger>
      <Collapsible.Content className="border-t border-[var(--border)] bg-[var(--surface)] px-3 pb-3 pt-2">
        <div className="mb-2">
          <div className="mb-1 text-xs font-medium text-[var(--text-muted)]">Arguments</div>
          <pre className="font-mono text-xs bg-[var(--bg)] rounded-[var(--radius-sm)] p-2 whitespace-pre-wrap break-all text-[var(--text)]">{tool.arguments}</pre>
        </div>
        {tool.status !== "running" && (
          <div>
            <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-[var(--text-muted)]">
              {tool.status === "error" ? "Error" : "Result"}
              <StatusIcon status={tool.status} />
            </div>
            <pre className="font-mono text-xs bg-[var(--bg)] rounded-[var(--radius-sm)] p-2 max-h-64 overflow-auto whitespace-pre-wrap break-all text-[var(--text-muted)]">
              {tool.status === "error" ? tool.error : tool.output}
            </pre>
          </div>
        )}
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
