import { useEffect, useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronRight, Circle, ListChecks } from "lucide-react";
import type { ReasoningStatus, ToolUse } from "../types";
import { cn } from "../lib/cn";
import { ToolCall } from "./ToolCall";

export function AgentActivity({
  reasoning,
  reasoningStatus,
  tools,
  messageStatus,
}: {
  reasoning: string;
  reasoningStatus: ReasoningStatus;
  tools: ToolUse[];
  messageStatus: "streaming" | "completed" | "failed" | "stopped" | "cancelled";
}) {
  const hasReasoning = Boolean(reasoning);
  const hasTools = tools.length > 0;
  const active =
    messageStatus === "streaming" ||
    reasoningStatus === "streaming" ||
    tools.some((tool) => tool.status === "running");
  const [open, setOpen] = useState(active);

  useEffect(() => {
    if (active) setOpen(true);
    else setOpen(false);
  }, [active]);

  if (!hasReasoning && !hasTools) return null;

  const failedTools = tools.filter((tool) => tool.status === "error").length;
  const runningTools = tools.filter((tool) => tool.status === "running").length;
  const toolSummary = hasTools
    ? `${tools.length} tool${tools.length === 1 ? "" : "s"}${
        runningTools ? ` running ${runningTools}` : ""
      }${failedTools ? ` failed ${failedTools}` : ""}`
    : "reasoning";
  const label = active ? "Working" : "Activity";

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen} className="mb-2 w-full">
      <Collapsible.Trigger className="group flex max-w-full items-center gap-2 py-1 text-xs text-[var(--text-faint)] hover:text-[var(--text-muted)] transition-colors">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center">
          <ChevronRight
            className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-90")}
          />
        </span>
        {active ? (
          <Circle className="h-2 w-2 shrink-0 fill-[var(--accent)] text-[var(--accent)] pulse-dot" />
        ) : (
          <ListChecks className="h-3.5 w-3.5 shrink-0 text-[var(--text-faint)] group-hover:text-[var(--text-muted)]" />
        )}
        <span className={cn("font-medium", active && "shimmer-text")}>
          {label}
        </span>
        <span className="text-[var(--text-faint)]">·</span>
        <span className="truncate">{toolSummary}</span>
      </Collapsible.Trigger>
      <Collapsible.Content className="ml-[9px] mt-1 border-l border-[var(--border)] pl-4">
        {hasReasoning && (
          <div className="mb-2 whitespace-pre-wrap text-xs leading-6 text-[var(--text-muted)]">
            {reasoning}
          </div>
        )}
        {tools.map((tool) => (
          <ToolCall key={tool.id} tool={tool} />
        ))}
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
