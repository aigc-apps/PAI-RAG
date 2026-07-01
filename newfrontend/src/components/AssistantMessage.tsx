import type { ChatMessage } from "../types";
import { Markdown } from "./Markdown";
import { MessageControls } from "./MessageControls";
import { AgentActivity } from "./AgentActivity";

export function AssistantMessage({
  message,
  onRegenerate,
}: {
  message: ChatMessage;
  onRegenerate?: () => void;
}) {
  const showControls =
    message.status === "completed" || message.status === "cancelled";
  return (
    <div className="flex gap-3 animate-msg-in">
      <div className="h-6 w-6 rounded-[var(--radius-sm)] shrink-0 mt-0.5 bg-[var(--surface-3)] flex items-center justify-center">
        <span className="text-xs font-bold text-[var(--text-muted)]">A</span>
      </div>
      <div className="flex-1 min-w-0">
        <AgentActivity
          reasoning={message.reasoning}
          reasoningStatus={message.reasoningStatus}
          tools={message.toolCalls}
          messageStatus={message.status}
        />
        {message.status === "failed" ? (
          <div className="rounded-[var(--radius-sm)] border border-[var(--danger)]/30 bg-[var(--danger)]/5 text-[var(--danger)] px-3 py-2 text-sm">
            {message.error || "Something went wrong."}
          </div>
        ) : (
          message.text && <Markdown content={message.text} />
        )}
        {message.status === "stopped" && (
          <div className="mt-1 text-xs text-[var(--text-faint)]">stopped</div>
        )}
        {message.status === "cancelled" && (
          <div className="mt-1 text-xs text-[var(--text-faint)]">cancelled</div>
        )}
        {showControls && (
          <MessageControls text={message.text} onRegenerate={onRegenerate} />
        )}
      </div>
    </div>
  );
}
