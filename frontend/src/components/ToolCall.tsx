import { useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { AlertTriangle, Check, X, ChevronRight } from "lucide-react";
import type { ToolUse } from "../types";
import { cn } from "../lib/cn";
import { getToolCallDisplay } from "../lib/toolCallDisplay";
import { useI18n, type MessageKey } from "../i18n";

function StatusDot({
  status,
  label,
}: {
  status: ToolUse["status"];
  label: string;
}) {
  const common = "h-2 w-2 shrink-0 rounded-full";
  if (status === "running")
    return (
      <span
        role="img"
        aria-label={label}
        className={`${common} bg-[var(--accent)] pulse-dot`}
      />
    );
  if (status === "done")
    return (
      <span
        role="img"
        aria-label={label}
        className={`${common} bg-[var(--success)]`}
      />
    );
  return (
    <span
      role="img"
      aria-label={label}
      className={`${common} bg-[var(--danger)]`}
    />
  );
}

function StatusIcon({ status }: { status: ToolUse["status"] }) {
  if (status === "done")
    return <Check className="h-3.5 w-3.5 text-[var(--success)]" />;
  if (status === "error")
    return <X className="h-3.5 w-3.5 text-[var(--danger)]" />;
  return null;
}

const STATUS_KEY: Record<ToolUse["status"], MessageKey> = {
  running: "tool.running",
  done: "tool.done",
  error: "tool.error",
};

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m${s}s`;
}

export function ToolCall({ tool }: { tool: ToolUse }) {
  const { t } = useI18n();
  // Errors auto-expand so the failure is visible without an extra click.
  const [open, setOpen] = useState(tool.status === "error");
  const display = getToolCallDisplay(tool.name, tool.arguments);

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
      <Collapsible.Trigger className="group flex w-full min-w-0 items-center gap-2 px-3 py-2 text-sm text-[var(--text-muted)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--text)]">
        <StatusDot status={tool.status} label={t(STATUS_KEY[tool.status])} />
        {display.labelKey ? (
          <>
            <span className="shrink-0 text-xs font-semibold text-[var(--text)]">
              {t(display.labelKey)}
            </span>
            <span className="shrink-0 font-mono text-xs text-[var(--text-muted)]">
              ({tool.name})
            </span>
          </>
        ) : (
          <span className="shrink-0 font-mono text-xs font-medium text-[var(--text)]">
            {tool.name}
          </span>
        )}
        {display.summary && (
          <>
            <span
              aria-hidden="true"
              className="shrink-0 text-xs text-[var(--text-faint)]"
            >
              ·
            </span>
            <span
              className="min-w-0 truncate font-mono text-xs text-[var(--text-faint)]"
              title={display.summary}
            >
              {display.summary}
            </span>
          </>
        )}
        <span
          data-testid="tool-duration"
          className="ml-auto w-14 shrink-0 text-right font-mono text-xs tabular-nums text-[var(--text-faint)]"
        >
          {tool.durationMs != null && tool.status !== "running"
            ? formatDuration(tool.durationMs)
            : null}
        </span>
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-[var(--text-faint)] transition-transform group-hover:text-[var(--text-muted)]",
            open && "rotate-90"
          )}
        />
      </Collapsible.Trigger>
      <Collapsible.Content className="border-t border-[var(--border)] bg-[var(--surface)] px-3 pb-3 pt-2">
        <div className="mb-2">
          <div className="mb-1 text-xs font-medium text-[var(--text-muted)]">{t("tool.arguments")}</div>
          <pre className="font-mono text-xs bg-[var(--bg)] rounded-[var(--radius-sm)] p-2 whitespace-pre-wrap break-all text-[var(--text)]">{tool.arguments}</pre>
        </div>
        {tool.status !== "running" && (
          <div>
            <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-[var(--text-muted)]">
              {t("tool.result")}
              <StatusIcon status={tool.status} />
            </div>
            {tool.status === "error" ? (
              <div className="flex gap-2 rounded-[var(--radius-sm)] border border-[var(--danger)]/30 bg-[var(--danger)]/5 px-2.5 py-2">
                <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0 text-[var(--danger)]" />
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-[var(--danger)]">{t("tool.execFailed")}</div>
                  <pre className="mt-0.5 font-mono text-xs whitespace-pre-wrap break-all text-[var(--text-muted)]">
                    {tool.error || t("tool.unknownError")}
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
