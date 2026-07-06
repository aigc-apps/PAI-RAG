import { AlertTriangle, RefreshCw } from "lucide-react";
import type { ChatMessage } from "../types";
import { deriveAssistantView } from "../stream/assistantView";
import { Markdown } from "./Markdown";
import { MessageControls } from "./MessageControls";
import { AgentActivity } from "./AgentActivity";
import { MessageArtifacts } from "./MessageArtifacts";

export function AssistantMessage({
  message,
  onRegenerate,
}: {
  message: ChatMessage;
  onRegenerate?: () => void;
}) {
  const showControls =
    message.status === "completed" || message.status === "cancelled";
  const failed = message.status === "failed";
  const { activitySteps, bodyText } = deriveAssistantView(message);
  const files = message.toolCalls.flatMap((t) => t.files ?? []);
  return (
    <div className="flex gap-3 animate-msg-in">
      <div className="h-6 w-6 rounded-[var(--radius-sm)] shrink-0 mt-0.5 bg-[var(--surface-3)] flex items-center justify-center">
        <span className="text-xs font-bold text-[var(--text-muted)]">A</span>
      </div>
      <div className="flex-1 min-w-0">
        <AgentActivity
          reasoning={message.reasoning}
          reasoningStatus={message.reasoningStatus}
          steps={activitySteps}
          messageStatus={message.status}
        />
        {failed ? (
          <div className="rounded-[var(--radius)] border border-[var(--danger)]/30 bg-[var(--danger)]/5 px-3.5 py-3">
            <div className="flex items-center gap-2 text-[var(--danger)]">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span className="text-sm font-semibold">执行失败</span>
            </div>
            <div className="mt-1.5 text-sm text-[var(--text-muted)] whitespace-pre-wrap">
              {message.error || "Something went wrong."}
            </div>
            {onRegenerate && (
              <button
                type="button"
                onClick={onRegenerate}
                className="mt-2.5 inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] border border-[var(--danger)]/40 bg-[var(--bg)] px-2.5 py-1.5 text-xs font-medium text-[var(--danger)] hover:bg-[var(--danger)]/10 transition-colors"
              >
                <RefreshCw className="h-3 w-3" />
                再试试
              </button>
            )}
          </div>
        ) : (
          bodyText && <Markdown content={bodyText} />
        )}
        {!failed && files.length > 0 && <MessageArtifacts files={files} />}
        {message.status === "stopped" && (
          <div className="mt-1 text-xs text-[var(--text-faint)]">已停止</div>
        )}
        {message.status === "cancelled" && (
          <div className="mt-1 text-xs text-[var(--text-faint)]">已取消</div>
        )}
        {showControls && (
          <MessageControls
            text={bodyText}
            usage={message.usage}
            onRegenerate={onRegenerate}
          />
        )}
      </div>
    </div>
  );
}
