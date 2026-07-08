import { useEffect, useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronRight, Circle, ListChecks, AlertTriangle } from "lucide-react";
import type { ReasoningStatus } from "../types";
import type { ResolvedStep } from "../stream/assistantView";
import { cn } from "../lib/cn";
import { ToolCall } from "./ToolCall";

export function AgentActivity({
  reasoning,
  reasoningStatus,
  steps,
  messageStatus,
}: {
  reasoning: string;
  reasoningStatus: ReasoningStatus;
  /** Ordered narration + tool timeline (the final answer is split out upstream). */
  steps: ResolvedStep[];
  messageStatus: "streaming" | "completed" | "failed" | "stopped" | "cancelled";
}) {
  const hasReasoning = Boolean(reasoning);
  const tools = steps.flatMap((s) => (s.kind === "tool" ? [s.tool] : []));
  const hasTools = tools.length > 0;
  const hasSteps = steps.length > 0;
  const active =
    messageStatus === "streaming" ||
    reasoningStatus === "streaming" ||
    tools.some((tool) => tool.status === "running");
  const [open, setOpen] = useState(active);

  useEffect(() => {
    if (active) setOpen(true);
    else setOpen(false);
  }, [active]);

  if (!hasReasoning && !hasSteps) return null;

  const failedTools = tools.filter((tool) => tool.status === "error").length;
  const runningTools = tools.filter((tool) => tool.status === "running").length;
  const doneTools = tools.filter((tool) => tool.status === "done").length;

  // A concise status chip summarising the tool run state.
  let badge: { label: string; tone: string } | null = null;
  if (failedTools > 0)
    badge = { label: `失败 ${failedTools}`, tone: "text-[var(--danger)]" };
  else if (runningTools > 0)
    badge = { label: "运行中", tone: "text-[var(--accent)]" };
  else if (hasTools)
    badge = { label: `完成 ${doneTools}/${tools.length}`, tone: "text-[var(--success)]" };

  const label = active ? "工作中" : "执行记录";

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen} className="mb-2 w-full">
      <Collapsible.Trigger
        className={cn(
          "group flex w-full items-center gap-2 rounded-[var(--radius-sm)] border px-2.5 py-1.5 text-xs transition-colors",
          failedTools > 0
            ? "border-[var(--danger)]/25 bg-[var(--danger)]/5 hover:bg-[var(--danger)]/10"
            : "border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-2)]"
        )}
      >
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-[var(--text-faint)] transition-transform group-hover:text-[var(--text-muted)]",
            open && "rotate-90"
          )}
        />
        {active ? (
          <Circle className="h-2 w-2 shrink-0 fill-[var(--accent)] text-[var(--accent)] pulse-dot" />
        ) : failedTools > 0 ? (
          <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-[var(--danger)]" />
        ) : (
          <ListChecks className="h-3.5 w-3.5 shrink-0 text-[var(--text-faint)] group-hover:text-[var(--text-muted)]" />
        )}
        <span className={cn("font-medium", active && "shimmer-text")}>{label}</span>
        <span className="text-[var(--text-faint)]">·</span>
        <span className="truncate text-[var(--text-muted)]">
          {hasTools ? `${tools.length} 个工具` : "推理"}
        </span>
        {badge && (
          <span className={cn("ml-auto shrink-0 text-xs font-semibold", badge.tone)}>
            {badge.label}
          </span>
        )}
      </Collapsible.Trigger>
      <Collapsible.Content className="ml-[9px] mt-1 border-l border-[var(--border)] pl-4">
        {hasReasoning && (
          <div className="mb-2 whitespace-pre-wrap text-xs leading-6 text-[var(--text-muted)]">
            {reasoning.trimEnd()}
          </div>
        )}
        {steps.map((step, i) =>
          step.kind === "tool" ? (
            // The HITL authorization card renders at the message level (always
            // visible), not here — this panel collapses/unmounts once the turn
            // ends. So the failing tool stays a plain ToolCall in the log.
            <ToolCall key={step.tool.id} tool={step.tool} />
          ) : (
            <div
              key={`text-${i}`}
              className="mb-2 whitespace-pre-wrap text-xs leading-6 text-[var(--text-muted)]"
            >
              {step.text.trimEnd()}
            </div>
          )
        )}
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
