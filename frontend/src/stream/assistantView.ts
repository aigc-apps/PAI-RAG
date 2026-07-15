import type { ChatMessage, ToolUse } from "../types";

/** A timeline step with its tool reference already resolved to the live object. */
export type ResolvedStep =
  | { kind: "reasoning"; text: string }
  | { kind: "text"; text: string }
  | { kind: "tool"; tool: ToolUse };

export interface AssistantView {
  /** Narration text + tool calls, in order, for the "working" panel. */
  activitySteps: ResolvedStep[];
  /** The final answer — the trailing text run — rendered as the message body. */
  bodyText: string;
}

/**
 * Split an assistant turn into its working timeline and its final answer.
 *
 * Live turns carry an ordered `steps` timeline: the last run of prose is the
 * answer; everything before it (interstitial narration + tool calls) is the
 * agent's working process and belongs in the collapsible activity panel.
 *
 * Reloaded history has no `steps` (persistence collapses the turn to one blob),
 * so we fall back to the legacy layout: the whole `text` is the body and every
 * tool call sits in the activity panel.
 */
export function deriveAssistantView(message: ChatMessage): AssistantView {
  const steps = message.steps ?? [];
  if (steps.length === 0) {
    return {
      activitySteps: message.toolCalls.map((tool) => ({ kind: "tool", tool })),
      bodyText: message.text,
    };
  }

  const toolById = new Map(message.toolCalls.map((t) => [t.id, t]));
  const lastIsText = steps[steps.length - 1].kind === "text";
  const bodyText = lastIsText
    ? (steps[steps.length - 1] as { text: string }).text
    : "";
  const activityRaw = lastIsText ? steps.slice(0, -1) : steps;

  const activitySteps: ResolvedStep[] = [];
  for (const step of activityRaw) {
    if (step.kind === "text" || step.kind === "reasoning") {
      // Drop whitespace-only narration runs (e.g. the trailing "\n\n" a model
      // emits before a tool call) so they don't render as empty paragraphs.
      if (step.text.trim()) activitySteps.push({ kind: step.kind, text: step.text });
    } else {
      const tool = toolById.get(step.id);
      if (tool) activitySteps.push({ kind: "tool", tool });
    }
  }
  return { activitySteps, bodyText };
}
