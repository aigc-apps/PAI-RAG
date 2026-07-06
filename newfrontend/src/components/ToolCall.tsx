import { useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { AlertTriangle, Check, X, ChevronRight } from "lucide-react";
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

const STATUS_WORD: Record<ToolUse["status"], string> = {
  running: "运行中",
  done: "完成",
  error: "失败",
};

const STATUS_TONE: Record<ToolUse["status"], string> = {
  running: "text-[var(--accent)]",
  done: "text-[var(--success)]",
  error: "text-[var(--danger)]",
};

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m${s}s`;
}

export function ToolCall({ tool }: { tool: ToolUse }) {
  // Errors auto-expand so the failure is visible without an extra click.
  const [open, setOpen] = useState(tool.status === "error");

  return (
    <Collapsible.Root
      open={open}
      onOpenChange={setOpen}
      className={cn(
        "my-1 overflow-hidden rounded-[var(--radius-sm)] border bg-transparent transition-colors",
        tool.status === "error"
          ? "border-[var(--danger)]/30 bg-[var(--danger)]/5"
          : "border-[var(--border)]"
      )}
    >
      <Collapsible.Trigger className="group w-full flex items-center gap-2 px-3 py-2 text-sm text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] transition-colors">
        <StatusDot status={tool.status} />
        <span className="font-mono text-xs font-medium text-[var(--text)]">{tool.name}</span>
        <span className={cn("text-xs font-medium ml-1", STATUS_TONE[tool.status])}>
          {STATUS_WORD[tool.status]}
        </span>
        {tool.durationMs != null && tool.status !== "running" && (
          <span className="text-xs text-[var(--text-faint)]">
            {formatDuration(tool.durationMs)}
          </span>
        )}
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 ml-auto shrink-0 text-[var(--text-faint)] transition-transform group-hover:text-[var(--text-muted)]",
            open && "rotate-90"
          )}
        />
      </Collapsible.Trigger>
      <Collapsible.Content className="border-t border-[var(--border)] bg-[var(--surface)] px-3 pb-3 pt-2">
        <div className="mb-2">
          <div className="mb-1 text-xs font-medium text-[var(--text-muted)]">参数</div>
          <pre className="font-mono text-xs bg-[var(--bg)] rounded-[var(--radius-sm)] p-2 whitespace-pre-wrap break-all text-[var(--text)]">{tool.arguments}</pre>
        </div>
        {tool.status !== "running" && (
          <div>
            <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-[var(--text-muted)]">
              {tool.status === "error" ? "结果" : "结果"}
              <StatusIcon status={tool.status} />
            </div>
            {tool.status === "error" ? (
              <div className="flex gap-2 rounded-[var(--radius-sm)] border border-[var(--danger)]/30 bg-[var(--danger)]/5 px-2.5 py-2">
                <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0 text-[var(--danger)]" />
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-[var(--danger)]">工具执行失败</div>
                  <pre className="mt-0.5 font-mono text-xs whitespace-pre-wrap break-all text-[var(--text-muted)]">
                    {tool.error || "Unknown error"}
                  </pre>
                </div>
              </div>
            ) : (
              <pre className="font-mono text-xs bg-[var(--bg)] rounded-[var(--radius-sm)] p-2 max-h-64 overflow-auto whitespace-pre-wrap break-all text-[var(--text-muted)]">
                {tool.output}
              </pre>
            )}
          </div>
        )}
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
